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

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"

namespace tensorrt_llm::executor
{

KvCacheConfig::KvCacheConfig(bool enableBlockReuse, std::optional<SizeType32> const& maxTokens,
    std::optional<std::vector<SizeType32>> const& maxAttentionWindowVec,
    std::optional<SizeType32> const& sinkTokenLength, std::optional<FloatType> const& freeGpuMemoryFraction,
    std::optional<size_t> const& hostCacheSize, bool onboardBlocks,
    std::optional<FloatType> const& crossKvCacheFraction, std::optional<RetentionPriority> secondaryOffloadMinPriority,
    size_t eventBufferMaxSize, std::optional<tensorrt_llm::runtime::RuntimeDefaults> const& runtimeDefaults,
    bool enablePartialReuse, bool copyOnPartialReuse,
    // Proactive KV cache configuration
    bool enableProactiveEviction, SizeType32 primaryFreeBlockThreshold,
    SizeType32 secondaryFreeBlockThreshold, SizeType32 proactiveEvictionBatchSize,
    SizeType32 minEvictionIntervalMs, SizeType32 maxWaitTimeMs,
    RetentionPriority evictionPriorityThreshold, bool enablePreloading,
    SizeType32 preloadBatchSize)
    : mEnableBlockReuse(enableBlockReuse)
    , mHostCacheSize(hostCacheSize)
    , mOnboardBlocks(onboardBlocks)
    , mSecondaryOffloadMinPriority(secondaryOffloadMinPriority)
    , mEventBufferMaxSize{eventBufferMaxSize}
    , mEnablePartialReuse{enablePartialReuse}
    , mCopyOnPartialReuse{copyOnPartialReuse}
    , mEnableProactiveEviction{enableProactiveEviction}
    , mPrimaryFreeBlockThreshold{primaryFreeBlockThreshold}
    , mSecondaryFreeBlockThreshold{secondaryFreeBlockThreshold}
    , mProactiveEvictionBatchSize{proactiveEvictionBatchSize}
    , mMinEvictionIntervalMs{minEvictionIntervalMs}
    , mMaxWaitTimeMs{maxWaitTimeMs}
    , mEvictionPriorityThreshold{evictionPriorityThreshold}
    , mEnablePreloading{enablePreloading}
    , mPreloadBatchSize{preloadBatchSize}
{
    if (maxTokens)
    {
        setMaxTokens(maxTokens.value());
    }
    if (maxAttentionWindowVec)
    {
        setMaxAttentionWindowVec(maxAttentionWindowVec.value());
    }
    if (sinkTokenLength)
    {
        setSinkTokenLength(sinkTokenLength.value());
    }
    if (freeGpuMemoryFraction)
    {
        setFreeGpuMemoryFraction(freeGpuMemoryFraction.value());
    }
    if (crossKvCacheFraction)
    {
        setCrossKvCacheFraction(crossKvCacheFraction.value());
    }
    if (runtimeDefaults)
    {
        fillEmptyFieldsFromRuntimeDefaults(runtimeDefaults.value());
    }
}

bool KvCacheConfig::getEnableBlockReuse() const
{
    return mEnableBlockReuse;
}

bool KvCacheConfig::getEnablePartialReuse() const
{
    return mEnablePartialReuse;
}

bool KvCacheConfig::getCopyOnPartialReuse() const
{
    return mCopyOnPartialReuse;
}

std::optional<SizeType32> KvCacheConfig::getMaxTokens() const
{
    return mMaxTokens;
}

std::optional<std::vector<SizeType32>> KvCacheConfig::getMaxAttentionWindowVec() const
{
    return mMaxAttentionWindowVec;
}

std::optional<SizeType32> KvCacheConfig::getSinkTokenLength() const
{
    return mSinkTokenLength;
}

std::optional<FloatType> KvCacheConfig::getFreeGpuMemoryFraction() const
{
    return mFreeGpuMemoryFraction;
}

std::optional<FloatType> KvCacheConfig::getCrossKvCacheFraction() const
{
    return mCrossKvCacheFraction;
}

std::optional<size_t> KvCacheConfig::getHostCacheSize() const
{
    return mHostCacheSize;
}

bool KvCacheConfig::getOnboardBlocks() const
{
    return mOnboardBlocks;
}

std::optional<RetentionPriority> KvCacheConfig::getSecondaryOffloadMinPriority() const
{
    return mSecondaryOffloadMinPriority;
}

size_t KvCacheConfig::getEventBufferMaxSize() const
{
    return mEventBufferMaxSize;
}

void KvCacheConfig::setEnableBlockReuse(bool enableBlockReuse)
{
    mEnableBlockReuse = enableBlockReuse;
}

void KvCacheConfig::setEnablePartialReuse(bool enablePartialReuse)
{
    mEnablePartialReuse = enablePartialReuse;
}

void KvCacheConfig::setCopyOnPartialReuse(bool copyOnPartialReuse)
{
    mCopyOnPartialReuse = copyOnPartialReuse;
}

void KvCacheConfig::setMaxTokens(SizeType32 maxTokens)
{
    TLLM_CHECK(maxTokens > 0);
    mMaxTokens = maxTokens;
}

void KvCacheConfig::setMaxAttentionWindowVec(std::vector<SizeType32> maxAttentionWindowVec)
{
    for (SizeType32 maxAttentionWindow : maxAttentionWindowVec)
    {
        TLLM_CHECK(maxAttentionWindow > 0);
    }
    mMaxAttentionWindowVec = maxAttentionWindowVec;
}

void KvCacheConfig::setSinkTokenLength(SizeType32 sinkTokenLength)
{
    TLLM_CHECK(sinkTokenLength > 0);
    mSinkTokenLength = sinkTokenLength;
}

void KvCacheConfig::setFreeGpuMemoryFraction(FloatType freeGpuMemoryFraction)
{
    TLLM_CHECK(freeGpuMemoryFraction > 0.F);
    TLLM_CHECK(freeGpuMemoryFraction < 1.F);
    mFreeGpuMemoryFraction = freeGpuMemoryFraction;
}

void KvCacheConfig::setCrossKvCacheFraction(FloatType crossKvCacheFraction)

{
    TLLM_CHECK(crossKvCacheFraction > 0.F);
    TLLM_CHECK(crossKvCacheFraction < 1.F);
    mCrossKvCacheFraction = crossKvCacheFraction;
}

void KvCacheConfig::setHostCacheSize(size_t hostCacheSize)
{
    mHostCacheSize = hostCacheSize;
}

void KvCacheConfig::setOnboardBlocks(bool onboardBlocks)
{
    mOnboardBlocks = onboardBlocks;
}

void KvCacheConfig::setSecondaryOffloadMinPriority(std::optional<RetentionPriority> secondaryOffloadMinPriority)
{
    mSecondaryOffloadMinPriority = secondaryOffloadMinPriority;
}

void KvCacheConfig::setEventBufferMaxSize(size_t eventBufferMaxSize)
{
    mEventBufferMaxSize = eventBufferMaxSize;
}

// Proactive KV cache configuration getters
bool KvCacheConfig::getEnableProactiveEviction() const
{
    return mEnableProactiveEviction;
}

SizeType32 KvCacheConfig::getPrimaryFreeBlockThreshold() const
{
    return mPrimaryFreeBlockThreshold;
}

SizeType32 KvCacheConfig::getSecondaryFreeBlockThreshold() const
{
    return mSecondaryFreeBlockThreshold;
}

SizeType32 KvCacheConfig::getProactiveEvictionBatchSize() const
{
    return mProactiveEvictionBatchSize;
}

SizeType32 KvCacheConfig::getMinEvictionIntervalMs() const
{
    return mMinEvictionIntervalMs;
}

SizeType32 KvCacheConfig::getMaxWaitTimeMs() const
{
    return mMaxWaitTimeMs;
}

RetentionPriority KvCacheConfig::getEvictionPriorityThreshold() const
{
    return mEvictionPriorityThreshold;
}

bool KvCacheConfig::getEnablePreloading() const
{
    return mEnablePreloading;
}

SizeType32 KvCacheConfig::getPreloadBatchSize() const
{
    return mPreloadBatchSize;
}

// Proactive KV cache configuration setters
void KvCacheConfig::setEnableProactiveEviction(bool enableProactiveEviction)
{
    mEnableProactiveEviction = enableProactiveEviction;
}

void KvCacheConfig::setPrimaryFreeBlockThreshold(SizeType32 primaryFreeBlockThreshold)
{
    mPrimaryFreeBlockThreshold = primaryFreeBlockThreshold;
}

void KvCacheConfig::setSecondaryFreeBlockThreshold(SizeType32 secondaryFreeBlockThreshold)
{
    mSecondaryFreeBlockThreshold = secondaryFreeBlockThreshold;
}

void KvCacheConfig::setProactiveEvictionBatchSize(SizeType32 proactiveEvictionBatchSize)
{
    mProactiveEvictionBatchSize = proactiveEvictionBatchSize;
}

void KvCacheConfig::setMinEvictionIntervalMs(SizeType32 minEvictionIntervalMs)
{
    mMinEvictionIntervalMs = minEvictionIntervalMs;
}

void KvCacheConfig::setMaxWaitTimeMs(SizeType32 maxWaitTimeMs)
{
    mMaxWaitTimeMs = maxWaitTimeMs;
}

void KvCacheConfig::setEvictionPriorityThreshold(RetentionPriority evictionPriorityThreshold)
{
    mEvictionPriorityThreshold = evictionPriorityThreshold;
}

void KvCacheConfig::setEnablePreloading(bool enablePreloading)
{
    mEnablePreloading = enablePreloading;
}

void KvCacheConfig::setPreloadBatchSize(SizeType32 preloadBatchSize)
{
    mPreloadBatchSize = preloadBatchSize;
}

void KvCacheConfig::fillEmptyFieldsFromRuntimeDefaults(tensorrt_llm::runtime::RuntimeDefaults runtimeDefaults)
{
    if (!mMaxAttentionWindowVec && runtimeDefaults.maxAttentionWindowVec)
    {
        setMaxAttentionWindowVec(runtimeDefaults.maxAttentionWindowVec.value());
    }
    if (!mSinkTokenLength && runtimeDefaults.sinkTokenLength)
    {
        setSinkTokenLength(runtimeDefaults.sinkTokenLength.value());
    }
}

} // namespace tensorrt_llm::executor
