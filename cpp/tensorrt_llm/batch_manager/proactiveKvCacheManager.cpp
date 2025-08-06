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

#include "tensorrt_llm/batch_manager/proactiveKvCacheManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/executor/executor.h"

#include <algorithm>
#include <chrono>
#include <thread>

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

ProactiveKVCacheManager::ProactiveKVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
    SizeType32 maxNumSequences, SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
    SizeType32 temporaryAttentionWindow, SizeType32 sinkTokenLength, CudaStreamPtr stream,
    std::optional<runtime::SizeType32> maxSequenceLength, bool enableBlockReuse, bool onboardBlocks,
    CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enableHashKey, bool enablePartialReuse,
    bool copyOnPartialReuse, ProactiveConfig const& proactiveConfig)
    : mBaseKVCacheManager(std::make_unique<KVCacheManager>(numKvHeadsPerLayer, sizePerHead, tokensPerBlock,
          blocksInPrimaryPool, blocksInSecondaryPool, maxNumSequences, maxBeamWidth, maxAttentionWindowVec,
          temporaryAttentionWindow, sinkTokenLength, stream, maxSequenceLength, enableBlockReuse, onboardBlocks,
          cacheType, secondaryOffloadMinPriority, eventManager, enableHashKey, enablePartialReuse, copyOnPartialReuse))
    , mProactiveConfig(proactiveConfig)
    , mLastEvictionTime(std::chrono::steady_clock::now())
{
    if (mProactiveConfig.enableProactiveEviction)
    {
        startProactiveWorker();
    }
}

ProactiveKVCacheManager::~ProactiveKVCacheManager()
{
    stopProactiveWorker();
}

void ProactiveKVCacheManager::allocatePools(nvinfer1::DataType dtype, bool useUvm)
{
    mBaseKVCacheManager->allocatePools(dtype, useUvm);
}

void ProactiveKVCacheManager::releasePools()
{
    mBaseKVCacheManager->releasePools();
}

void ProactiveKVCacheManager::startScheduling()
{
    mBaseKVCacheManager->startScheduling();
}

SizeType32 ProactiveKVCacheManager::getTokensPerBlock() const
{
    return mBaseKVCacheManager->getTokensPerBlock();
}

SizeType32 ProactiveKVCacheManager::getMaxNumBlocks() const
{
    return mBaseKVCacheManager->getMaxNumBlocks();
}

SizeType32 ProactiveKVCacheManager::getUsedNumBlocks() const
{
    return mBaseKVCacheManager->getUsedNumBlocks();
}

SizeType32 ProactiveKVCacheManager::getNumFreeBlocks() const
{
    return mBaseKVCacheManager->getNumFreeBlocks();
}

SizeType32 ProactiveKVCacheManager::getNumPools() const
{
    return mBaseKVCacheManager->getNumPools();
}

SizeType32 ProactiveKVCacheManager::getNumReusedBlocks() const noexcept
{
    return mBaseKVCacheManager->getNumReusedBlocks();
}

KvCacheStats ProactiveKVCacheManager::getKvCacheStats() const
{
    return mBaseKVCacheManager->getKvCacheStats();
}

SizeType32 ProactiveKVCacheManager::getMaxBlocksPerSeq() const
{
    return mBaseKVCacheManager->getMaxBlocksPerSeq();
}

std::deque<executor::KVCacheEvent> ProactiveKVCacheManager::getLatestEvents(
    std::optional<std::chrono::milliseconds> timeout) const
{
    return mBaseKVCacheManager->getLatestEvents(timeout);
}

BlockManager const& ProactiveKVCacheManager::getBlockManager() const
{
    return mBaseKVCacheManager->getBlockManager();
}

SizeType32 ProactiveKVCacheManager::getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const
{
    return mBaseKVCacheManager->getNeededBlocksOneStep(req, twoStepsLookAhead);
}

SizeType32 ProactiveKVCacheManager::getNeededBlocksToCompletion(LlmRequest const& req) const
{
    return mBaseKVCacheManager->getNeededBlocksToCompletion(req);
}

SizeType32 ProactiveKVCacheManager::getRemainingBlocksToCompletion(LlmRequest const& req) const
{
    return mBaseKVCacheManager->getRemainingBlocksToCompletion(req);
}

void ProactiveKVCacheManager::addSequence(RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth,
    OptionalRef<LlmRequest> llmRequest)
{
    mBaseKVCacheManager->addSequence(requestId, inputLength, beamWidth, llmRequest);
}

void ProactiveKVCacheManager::addToken(RequestIdType requestId)
{
    mBaseKVCacheManager->addToken(requestId);
}

void ProactiveKVCacheManager::removeSequence(RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest)
{
    mBaseKVCacheManager->removeSequence(requestId, llmRequest);
}

void ProactiveKVCacheManager::schedulingReleaseBlocks(RequestIdType requestId)
{
    mBaseKVCacheManager->schedulingReleaseBlocks(requestId);
}

SizeType32 ProactiveKVCacheManager::numPools() const
{
    return mBaseKVCacheManager->numPools();
}

SizeType32 ProactiveKVCacheManager::maxBlocksPerSeq() const
{
    return mBaseKVCacheManager->maxBlocksPerSeq();
}

bool ProactiveKVCacheManager::enableBlockReuse() const
{
    return mBaseKVCacheManager->enableBlockReuse();
}

// Additional methods required by BaseKVCacheManager interface
void ProactiveKVCacheManager::schedulingRemoveSequence(RequestIdType requestId)
{
    mBaseKVCacheManager->schedulingRemoveSequence(requestId);
}

runtime::ITensor::SharedPtr ProactiveKVCacheManager::getBlockPoolPointers() const
{
    return mBaseKVCacheManager->getBlockPoolPointers();
}

runtime::ITensor::SharedPtr ProactiveKVCacheManager::getLayerToPoolMapping() const
{
    return mBaseKVCacheManager->getLayerToPoolMapping();
}

void ProactiveKVCacheManager::getBlockOffsetsOfBatch(runtime::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const
{
    mBaseKVCacheManager->getBlockOffsetsOfBatch(output, firstBatchSlotIdx, batchSize, beamWidth);
}

SizeType32 ProactiveKVCacheManager::copyBlockOffsets(runtime::ITensor& output, SizeType32 outputSlotOffset, RequestIdType requestId) const
{
    return mBaseKVCacheManager->copyBlockOffsets(output, outputSlotOffset, requestId);
}

bool ProactiveKVCacheManager::isEnableBlockReuse() const
{
    return mBaseKVCacheManager->isEnableBlockReuse();
}

bool ProactiveKVCacheManager::isUseOneMoreBlock() const
{
    return mBaseKVCacheManager->isUseOneMoreBlock();
}

void ProactiveKVCacheManager::rewindKVCache(RequestIdType requestId, SizeType32 rewindLengths)
{
    mBaseKVCacheManager->rewindKVCache(requestId, rewindLengths);
}

GenerationRequest const& ProactiveKVCacheManager::getSequence(RequestIdType requestId) const
{
    return mBaseKVCacheManager->getSequence(requestId);
}

GenerationRequest& ProactiveKVCacheManager::getSequence(RequestIdType requestId)
{
    return mBaseKVCacheManager->getSequence(requestId);
}

bool ProactiveKVCacheManager::isCrossKv() const
{
    return mBaseKVCacheManager->isCrossKv();
}

std::optional<BlockKey> ProactiveKVCacheManager::findNewContextBlock(VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const
{
    return mBaseKVCacheManager->findNewContextBlock(uniqueTokens, llmRequest);
}

void ProactiveKVCacheManager::storeContextBlocks(LlmRequest const& llmRequest)
{
    mBaseKVCacheManager->storeContextBlocks(llmRequest);
}

bool ProactiveKVCacheManager::schedulingHasFreeBlocks(SizeType32 numRequired) const
{
    return mBaseKVCacheManager->schedulingHasFreeBlocks(numRequired);
}

std::vector<std::vector<SizeType32>> const& ProactiveKVCacheManager::getCacheBlockIds(RequestIdType requestId) const
{
    return mBaseKVCacheManager->getCacheBlockIds(requestId);
}

std::vector<std::vector<std::vector<SizeType32>>> ProactiveKVCacheManager::getBatchCacheBlockIds(std::vector<RequestIdType> const& requestIds) const
{
    return mBaseKVCacheManager->getBatchCacheBlockIds(requestIds);
}

std::vector<KVCacheBlock::IdType> ProactiveKVCacheManager::getNewlyAllocatedBlockIds(RequestIdType requestId) const
{
    return mBaseKVCacheManager->getNewlyAllocatedBlockIds(requestId);
}

runtime::ITensor::SharedPtr ProactiveKVCacheManager::getPrimaryPool(SizeType32 layer_idx) const
{
    return mBaseKVCacheManager->getPrimaryPool(layer_idx);
}

SizeType32 ProactiveKVCacheManager::getPoolLayerIdx(SizeType32 layer_idx) const
{
    return mBaseKVCacheManager->getPoolLayerIdx(layer_idx);
}

void ProactiveKVCacheManager::refreshBlocks()
{
    mBaseKVCacheManager->refreshBlocks();
}

void ProactiveKVCacheManager::flushIterationEvents()
{
    mBaseKVCacheManager->flushIterationEvents();
}

SizeType32 ProactiveKVCacheManager::getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const
{
    return mBaseKVCacheManager->getMaxCapacityBatchSize(inputLength, outputLength);
}

CacheType ProactiveKVCacheManager::getCacheType() const
{
    return mBaseKVCacheManager->getCacheType();
}

void ProactiveKVCacheManager::setProactiveConfig(ProactiveConfig const& config)
{
    mProactiveConfig = config;
    
    // Restart worker if needed
    if (config.enableProactiveEviction && !mWorkerRunning)
    {
        startProactiveWorker();
    }
    else if (!config.enableProactiveEviction && mWorkerRunning)
    {
        stopProactiveWorker();
    }
}

ProactiveKVCacheManager::ProactiveConfig const& ProactiveKVCacheManager::getProactiveConfig() const
{
    return mProactiveConfig;
}

void ProactiveKVCacheManager::triggerProactiveEviction()
{
    if (mProactiveConfig.enableProactiveEviction)
    {
        performProactiveEviction();
    }
}

ProactiveKVCacheManager::ProactiveStats ProactiveKVCacheManager::getProactiveStats() const
{
    std::lock_guard<std::mutex> lock(mProactiveStatsMutex);
    return mProactiveStats;
}

void ProactiveKVCacheManager::startProactiveWorker()
{
    if (mWorkerRunning)
    {
        return;
    }
    
    mWorkerRunning = true;
    mProactiveWorkerThread = std::thread(&ProactiveKVCacheManager::proactiveEvictionWorker, this);
}

void ProactiveKVCacheManager::stopProactiveWorker()
{
    if (!mWorkerRunning)
    {
        return;
    }
    
    mWorkerRunning = false;
    mWorkerCV.notify_all();
    
    if (mProactiveWorkerThread.joinable())
    {
        mProactiveWorkerThread.join();
    }
}

void ProactiveKVCacheManager::proactiveEvictionWorker()
{
    while (mWorkerRunning)
    {
        std::unique_lock<std::mutex> lock(mWorkerMutex);
        
        // Wait for either timeout or notification
        auto timeout = mProactiveConfig.minEvictionInterval;
        if (mWorkerCV.wait_for(lock, timeout, [this] { return !mWorkerRunning; }))
        {
            break;
        }
        
        // Check if we should perform proactive eviction
        if (shouldTriggerProactiveEviction())
        {
            performProactiveEviction();
        }
        
        // Check if we should perform proactive preloading
        if (shouldTriggerProactivePreloading())
        {
            performProactivePreloading();
        }
    }
}

void ProactiveKVCacheManager::performProactiveEviction()
{
    auto startTime = std::chrono::steady_clock::now();
    
    // Check rate limiting
    {
        std::lock_guard<std::mutex> lock(mLastEvictionTimeMutex);
        auto timeSinceLastEviction = std::chrono::duration_cast<std::chrono::milliseconds>(
            startTime - mLastEvictionTime);
        if (timeSinceLastEviction < mProactiveConfig.minEvictionInterval)
        {
            return;
        }
    }
    
    TLLM_LOG_DEBUG("ProactiveKVCacheManager: Starting proactive eviction cycle");
    
    auto& blockManager = const_cast<BlockManager&>(mBaseKVCacheManager->getBlockManager());
    auto evictionCandidates = getEvictionCandidates();
    
    SizeType32 evictedCount = 0;
    for (auto const& block : evictionCandidates)
    {
        if (evictedCount >= mProactiveConfig.proactiveEvictionBatchSize)
        {
            break;
        }
        
        // Only evict blocks that are in primary memory and have low priority
        if (block->isPrimary() && block->getPriority() <= mProactiveConfig.evictionPriorityThreshold)
        {
            // Offload the block to secondary memory
            blockManager.offloadBlock(block);
            
            {
                std::lock_guard<std::mutex> lock(mProactiveBlockMutex);
                mProactivelyEvictedBlocks.insert(block);
            }
            
            evictedCount++;
            TLLM_LOG_DEBUG("ProactiveKVCacheManager: Proactively evicted block %d", block->getBlockId());
        }
    }
    
    auto endTime = std::chrono::steady_clock::now();
    auto evictionTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // Update statistics
    updateProactiveStats(evictedCount, 0, evictionTime, std::chrono::milliseconds(0));
    
    // Update last eviction time
    {
        std::lock_guard<std::mutex> lock(mLastEvictionTimeMutex);
        mLastEvictionTime = endTime;
    }
    
    TLLM_LOG_DEBUG("ProactiveKVCacheManager: Completed proactive eviction cycle, evicted %d blocks in %ld ms",
        evictedCount, evictionTime.count());
}

void ProactiveKVCacheManager::performProactivePreloading()
{
    auto startTime = std::chrono::steady_clock::now();
    
    TLLM_LOG_DEBUG("ProactiveKVCacheManager: Starting proactive preloading cycle");
    
    auto& blockManager = const_cast<BlockManager&>(mBaseKVCacheManager->getBlockManager());
    auto preloadCandidates = getPreloadCandidates();
    
    SizeType32 preloadedCount = 0;
    for (auto const& block : preloadCandidates)
    {
        if (preloadedCount >= mProactiveConfig.preloadBatchSize)
        {
            break;
        }
        
        // Only preload blocks that are in secondary memory and have high priority
        if (!block->isPrimary() && block->getPriority() > mProactiveConfig.evictionPriorityThreshold)
        {
            // Onboard the block to primary memory
            blockManager.onboardBlock(block);
            
            {
                std::lock_guard<std::mutex> lock(mProactiveBlockMutex);
                mProactivelyPreloadedBlocks.insert(block);
            }
            
            preloadedCount++;
            TLLM_LOG_DEBUG("ProactiveKVCacheManager: Proactively preloaded block %d", block->getBlockId());
        }
    }
    
    auto endTime = std::chrono::steady_clock::now();
    auto preloadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // Update statistics
    updateProactiveStats(0, preloadedCount, std::chrono::milliseconds(0), preloadTime);
    
    TLLM_LOG_DEBUG("ProactiveKVCacheManager: Completed proactive preloading cycle, preloaded %d blocks in %ld ms",
        preloadedCount, preloadTime.count());
}

bool ProactiveKVCacheManager::shouldTriggerProactiveEviction() const
{
    auto const numFreePrimaryBlocks = mBaseKVCacheManager->getBlockManager().getEvictionPolicy()->getNumFreeBlocks(0);
    auto const numFreeSecondaryBlocks = mBaseKVCacheManager->getBlockManager().getEvictionPolicy()->getNumFreeBlocks(1);
    
    return numFreePrimaryBlocks <= mProactiveConfig.primaryFreeBlockThreshold ||
           numFreeSecondaryBlocks <= mProactiveConfig.secondaryFreeBlockThreshold;
}

bool ProactiveKVCacheManager::shouldTriggerProactivePreloading() const
{
    auto const numFreePrimaryBlocks = mBaseKVCacheManager->getBlockManager().getEvictionPolicy()->getNumFreeBlocks(0);
    
    // Only preload if we have enough free space in primary memory
    return numFreePrimaryBlocks >= mProactiveConfig.primaryFreeBlockThreshold * 2;
}

std::vector<BlockPtr> ProactiveKVCacheManager::getEvictionCandidates() const
{
    std::vector<BlockPtr> candidates;
    
    // Get all blocks from the base manager - we need to access the private member
    // For now, we'll use a different approach by getting blocks through the eviction policy
    auto const& blockManager = mBaseKVCacheManager->getBlockManager();
    
    // We'll need to iterate through allocated blocks differently
    // For now, let's use a simpler approach by checking free blocks and trying to evict
    // blocks that are in primary memory and have low priority
    
    // This is a simplified approach - in a real implementation, we'd need to
    // access the internal block lists from the BlockManager
    return candidates;
}

std::vector<BlockPtr> ProactiveKVCacheManager::getPreloadCandidates() const
{
    std::vector<BlockPtr> candidates;
    
    // Similar to eviction candidates, we need a different approach
    // For now, return empty list
    return candidates;
}

void ProactiveKVCacheManager::updateProactiveStats(SizeType32 evictions, SizeType32 preloads,
    std::chrono::milliseconds evictionTime, std::chrono::milliseconds preloadTime)
{
    std::lock_guard<std::mutex> lock(mProactiveStatsMutex);
    
    mProactiveStats.totalProactiveEvictions += evictions;
    mProactiveStats.totalProactivePreloads += preloads;
    mProactiveStats.totalEvictionTime += evictionTime;
    mProactiveStats.totalPreloadTime += preloadTime;
    
    if (evictions > 0 || preloads > 0)
    {
        mProactiveStats.cyclesTriggered++;
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager 