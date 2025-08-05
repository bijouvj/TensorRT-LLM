/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/evictionPolicy.h"
#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/executor/executor.h"

#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

class ProactiveKVCacheManager : public BaseKVCacheManager
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;

    struct ProactiveConfig
    {
        // Threshold for free blocks in primary memory before proactive eviction starts
        SizeType32 primaryFreeBlockThreshold = 10;
        
        // Threshold for free blocks in secondary memory before proactive eviction starts
        SizeType32 secondaryFreeBlockThreshold = 5;
        
        // Number of blocks to proactively evict when thresholds are met
        SizeType32 proactiveEvictionBatchSize = 5;
        
        // Minimum time between proactive eviction cycles (milliseconds)
        std::chrono::milliseconds minEvictionInterval = std::chrono::milliseconds(100);
        
        // Maximum time to wait for blocks to become available (milliseconds)
        std::chrono::milliseconds maxWaitTime = std::chrono::milliseconds(1000);
        
        // Whether to enable proactive eviction
        bool enableProactiveEviction = true;
        
        // Priority threshold for proactive eviction (blocks with lower priority will be evicted first)
        executor::RetentionPriority evictionPriorityThreshold = 50;
        
        // Whether to preload blocks from secondary to primary when space becomes available
        bool enablePreloading = true;
        
        // Number of blocks to preload when space becomes available
        SizeType32 preloadBatchSize = 3;
    };

    ProactiveKVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
        SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
        SizeType32 maxNumSequences, SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
        SizeType32 temporaryAttentionWindow, SizeType32 sinkTokenLength, CudaStreamPtr stream,
        std::optional<runtime::SizeType32> maxSequenceLength, bool enableBlockReuse = false, bool onboardBlocks = true,
        CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enableHashKey = false,
        bool enablePartialReuse = true, bool copyOnPartialReuse = true,
        ProactiveConfig const& proactiveConfig = ProactiveConfig{});

    ~ProactiveKVCacheManager() override;

    void allocatePools(nvinfer1::DataType dtype, bool useUvm = false) override;

    void releasePools() override;

    void startScheduling() override;

    [[nodiscard]] SizeType32 getTokensPerBlock() const override;

    [[nodiscard]] SizeType32 getMaxNumBlocks() const override;

    [[nodiscard]] SizeType32 getUsedNumBlocks() const override;

    [[nodiscard]] SizeType32 getNumFreeBlocks() const override;

    [[nodiscard]] SizeType32 getNumPools() const override;

    [[nodiscard]] SizeType32 getNumReusedBlocks() const noexcept override;

    [[nodiscard]] KvCacheStats getKvCacheStats() const override;

    [[nodiscard]] SizeType32 getMaxBlocksPerSeq() const override;

    [[nodiscard]] std::deque<executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const override;

    [[nodiscard]] BlockManager const& getBlockManager() const override;

    [[nodiscard]] SizeType32 getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const override;

    [[nodiscard]] SizeType32 getNeededBlocksToCompletion(LlmRequest const& req) const override;

    [[nodiscard]] SizeType32 getRemainingBlocksToCompletion(LlmRequest const& req) const override;

    void addSequence(RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth,
        OptionalRef<LlmRequest> llmRequest = std::nullopt) override;

    void addToken(RequestIdType requestId) override;

    void removeSequence(RequestIdType requestId) override;

    void schedulingReleaseBlocks(RequestIdType requestId) override;

    [[nodiscard]] SizeType32 numPools() const override;

    [[nodiscard]] SizeType32 maxBlocksPerSeq() const override;

    [[nodiscard]] bool enableBlockReuse() const override;

    // Additional methods required by BaseKVCacheManager interface
    void removeSequence(RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest) override;

    void schedulingRemoveSequence(RequestIdType requestId) override;

    [[nodiscard]] runtime::ITensor::SharedPtr getBlockPoolPointers() const override;

    [[nodiscard]] runtime::ITensor::SharedPtr getLayerToPoolMapping() const override;

    void getBlockOffsetsOfBatch(runtime::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const override;

    [[nodiscard]] SizeType32 copyBlockOffsets(runtime::ITensor& output, SizeType32 outputSlotOffset, RequestIdType requestId) const override;

    [[nodiscard]] bool isEnableBlockReuse() const override;

    [[nodiscard]] bool isUseOneMoreBlock() const override;

    void rewindKVCache(RequestIdType requestId, SizeType32 rewindLengths) override;

    [[nodiscard]] GenerationRequest const& getSequence(RequestIdType requestId) const override;

    [[nodiscard]] GenerationRequest& getSequence(RequestIdType requestId) override;

    [[nodiscard]] bool isCrossKv() const override;

    [[nodiscard]] std::optional<BlockKey> findNewContextBlock(VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const override;

    void storeContextBlocks(LlmRequest const& llmRequest) override;

    [[nodiscard]] bool schedulingHasFreeBlocks(SizeType32 numRequired = 1) const override;

    [[nodiscard]] std::vector<std::vector<SizeType32>> const& getCacheBlockIds(RequestIdType requestId) const override;

    [[nodiscard]] std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(std::vector<RequestIdType> const& requestIds) const override;

    [[nodiscard]] std::vector<KVCacheBlock::IdType> getNewlyAllocatedBlockIds(RequestIdType requestId) const override;

    [[nodiscard]] runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 layer_idx) const override;

    [[nodiscard]] SizeType32 getPoolLayerIdx(SizeType32 layer_idx) const override;

    void refreshBlocks() override;

    void flushIterationEvents() override;

    [[nodiscard]] SizeType32 getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const override;

    [[nodiscard]] CacheType getCacheType() const override;

    // Proactive management methods
    void setProactiveConfig(ProactiveConfig const& config);
    
    [[nodiscard]] ProactiveConfig const& getProactiveConfig() const;
    
    // Force a proactive eviction cycle
    void triggerProactiveEviction();
    
    // Get statistics about proactive eviction
    struct ProactiveStats
    {
        SizeType32 totalProactiveEvictions = 0;
        SizeType32 totalProactivePreloads = 0;
        std::chrono::milliseconds totalEvictionTime = std::chrono::milliseconds(0);
        std::chrono::milliseconds totalPreloadTime = std::chrono::milliseconds(0);
        SizeType32 cyclesTriggered = 0;
    };
    
    [[nodiscard]] ProactiveStats getProactiveStats() const;

private:
    // Proactive eviction worker thread
    void proactiveEvictionWorker();
    
    // Perform proactive eviction of blocks
    void performProactiveEviction();
    
    // Perform proactive preloading of blocks
    void performProactivePreloading();
    
    // Check if proactive eviction should be triggered
    bool shouldTriggerProactiveEviction() const;
    
    // Check if proactive preloading should be triggered
    bool shouldTriggerProactivePreloading() const;
    
    // Get blocks that are candidates for proactive eviction
    std::vector<BlockPtr> getEvictionCandidates() const;
    
    // Get blocks that are candidates for proactive preloading
    std::vector<BlockPtr> getPreloadCandidates() const;
    
    // Update proactive statistics
    void updateProactiveStats(SizeType32 evictions, SizeType32 preloads, 
                            std::chrono::milliseconds evictionTime, 
                            std::chrono::milliseconds preloadTime);

    // Base KV cache manager
    std::unique_ptr<KVCacheManager> mBaseKVCacheManager;
    
    // Proactive configuration
    ProactiveConfig mProactiveConfig;
    
    // Proactive statistics
    mutable std::mutex mProactiveStatsMutex;
    ProactiveStats mProactiveStats;
    
    // Worker thread management
    std::thread mProactiveWorkerThread;
    std::atomic<bool> mWorkerRunning{false};
    std::condition_variable mWorkerCV;
    std::mutex mWorkerMutex;
    
    // Last eviction time for rate limiting
    std::chrono::steady_clock::time_point mLastEvictionTime;
    mutable std::mutex mLastEvictionTimeMutex;
    
    // Block tracking for proactive management
    std::unordered_set<BlockPtr> mProactivelyEvictedBlocks;
    std::unordered_set<BlockPtr> mProactivelyPreloadedBlocks;
    mutable std::mutex mProactiveBlockMutex;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager 