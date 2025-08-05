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
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/executor/executor.h"

#include <iostream>
#include <memory>
#include <chrono>
#include <thread>

using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::runtime;

int main()
{
    std::cout << "Proactive KV Cache Manager Example" << std::endl;
    std::cout << "==================================" << std::endl;

    // Configuration parameters
    std::vector<SizeType32> numKvHeadsPerLayer = {32, 32, 32, 32, 32, 32}; // 6 layers, 32 heads each
    SizeType32 sizePerHead = 128;
    SizeType32 tokensPerBlock = 64;
    SizeType32 blocksInPrimaryPool = 1000;
    SizeType32 blocksInSecondaryPool = 2000;
    SizeType32 maxNumSequences = 100;
    SizeType32 maxBeamWidth = 1;
    std::vector<SizeType32> maxAttentionWindowVec = {2048};
    SizeType32 temporaryAttentionWindow = 0;
    SizeType32 sinkTokenLength = 0;
    
    // Create CUDA stream
    auto stream = std::make_shared<CudaStream>();
    
    // Proactive configuration
    ProactiveKVCacheManager::ProactiveConfig proactiveConfig;
    proactiveConfig.primaryFreeBlockThreshold = 50;      // Start eviction when < 50 free blocks in primary
    proactiveConfig.secondaryFreeBlockThreshold = 100;   // Start eviction when < 100 free blocks in secondary
    proactiveConfig.proactiveEvictionBatchSize = 10;     // Evict 10 blocks at a time
    proactiveConfig.minEvictionInterval = std::chrono::milliseconds(200); // Minimum 200ms between evictions
    proactiveConfig.maxWaitTime = std::chrono::milliseconds(1000);
    proactiveConfig.enableProactiveEviction = true;
    proactiveConfig.evictionPriorityThreshold = 50;      // Evict blocks with priority <= 50
    proactiveConfig.enablePreloading = true;
    proactiveConfig.preloadBatchSize = 5;                // Preload 5 blocks at a time

    // Create proactive KV cache manager
    auto proactiveKvCacheManager = std::make_unique<ProactiveKVCacheManager>(
        numKvHeadsPerLayer, sizePerHead, tokensPerBlock, blocksInPrimaryPool, blocksInSecondaryPool,
        maxNumSequences, maxBeamWidth, maxAttentionWindowVec, temporaryAttentionWindow, sinkTokenLength,
        stream, std::nullopt, true, true, CacheType::kSELF, std::nullopt, nullptr, false, true, true,
        proactiveConfig);

    // Allocate pools
    proactiveKvCacheManager->allocatePools(nvinfer1::DataType::kHALF, false);

    std::cout << "Initial KV cache statistics:" << std::endl;
    std::cout << "  Max blocks: " << proactiveKvCacheManager->getMaxNumBlocks() << std::endl;
    std::cout << "  Free blocks: " << proactiveKvCacheManager->getNumFreeBlocks() << std::endl;
    std::cout << "  Used blocks: " << proactiveKvCacheManager->getUsedNumBlocks() << std::endl;
    std::cout << "  Tokens per block: " << proactiveKvCacheManager->getTokensPerBlock() << std::endl;

    // Simulate some sequences
    std::vector<RequestIdType> requestIds = {1, 2, 3, 4, 5};
    std::vector<SizeType32> inputLengths = {512, 1024, 768, 1536, 256};

    std::cout << "\nAdding sequences..." << std::endl;
    for (size_t i = 0; i < requestIds.size(); ++i)
    {
        proactiveKvCacheManager->addSequence(requestIds[i], inputLengths[i], maxBeamWidth);
        std::cout << "  Added sequence " << requestIds[i] << " with " << inputLengths[i] << " tokens" << std::endl;
    }

    std::cout << "\nAfter adding sequences:" << std::endl;
    std::cout << "  Free blocks: " << proactiveKvCacheManager->getNumFreeBlocks() << std::endl;
    std::cout << "  Used blocks: " << proactiveKvCacheManager->getUsedNumBlocks() << std::endl;

    // Simulate token generation
    std::cout << "\nSimulating token generation..." << std::endl;
    for (int step = 0; step < 10; ++step)
    {
        for (auto requestId : requestIds)
        {
            proactiveKvCacheManager->addToken(requestId);
        }
        
        if (step % 5 == 0)
        {
            std::cout << "  Step " << step << ": Free blocks = " << proactiveKvCacheManager->getNumFreeBlocks() << std::endl;
            
            // Get proactive statistics
            auto stats = proactiveKvCacheManager->getProactiveStats();
            std::cout << "    Proactive evictions: " << stats.totalProactiveEvictions << std::endl;
            std::cout << "    Proactive preloads: " << stats.totalProactivePreloads << std::endl;
            std::cout << "    Cycles triggered: " << stats.cyclesTriggered << std::endl;
        }
    }

    // Force a proactive eviction cycle
    std::cout << "\nForcing proactive eviction cycle..." << std::endl;
    proactiveKvCacheManager->triggerProactiveEviction();

    // Get final statistics
    auto finalStats = proactiveKvCacheManager->getProactiveStats();
    std::cout << "\nFinal proactive statistics:" << std::endl;
    std::cout << "  Total proactive evictions: " << finalStats.totalProactiveEvictions << std::endl;
    std::cout << "  Total proactive preloads: " << finalStats.totalProactivePreloads << std::endl;
    std::cout << "  Total eviction time: " << finalStats.totalEvictionTime.count() << " ms" << std::endl;
    std::cout << "  Total preload time: " << finalStats.totalPreloadTime.count() << " ms" << std::endl;
    std::cout << "  Cycles triggered: " << finalStats.cyclesTriggered << std::endl;

    // Remove sequences
    std::cout << "\nRemoving sequences..." << std::endl;
    for (auto requestId : requestIds)
    {
        proactiveKvCacheManager->removeSequence(requestId);
        std::cout << "  Removed sequence " << requestId << std::endl;
    }

    std::cout << "\nFinal KV cache statistics:" << std::endl;
    std::cout << "  Free blocks: " << proactiveKvCacheManager->getNumFreeBlocks() << std::endl;
    std::cout << "  Used blocks: " << proactiveKvCacheManager->getUsedNumBlocks() << std::endl;

    // Release pools
    proactiveKvCacheManager->releasePools();

    std::cout << "\nExample completed successfully!" << std::endl;
    return 0;
} 