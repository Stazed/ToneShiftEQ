
/*
 * InstanceRegistry.h
 *
 * SPDX-License-Identifier:  BSD-3-Clause
 *
 * Copyright (C) 2026 brummer <brummer@web.de>
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>
#include <string>

class InstanceRegistry {
public:
    using InstanceID = uint32_t;

    struct Instance {
        InstanceID id;
        std::string name;
        void* ptr;
    };

    static InstanceRegistry& instance() {
        static InstanceRegistry registry;
        return registry;
    }

    InstanceID registerInstance(void* ptr) {
        if (!ptr) return 0;

        std::lock_guard<std::mutex> lock(mutex_);
        // Don't register the same instance twice.
        for (const auto& instance : instances_) {
            if (instance.ptr == ptr) {
                return instance.id;
            }
        }

        const InstanceID id = nextId_++;
        std::string name =  "#" + std::to_string(id);
        instances_.push_back({id, name, ptr});

        return id;
    }

    void registerInstanceName(InstanceID id, std::string instanceName) {
        for (auto& instance : instances_) {
            if (instance.id == id) {
                instance.name = instanceName;
                break;
            }
        }        
    }

    void unregisterInstance(void* ptr) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex_);

        instances_.erase(
            std::remove_if(instances_.begin(), instances_.end(),
                [ptr](const Instance& instance)
                {
                    return instance.ptr == ptr;
                }),
            instances_.end());
    }

    std::vector<Instance> getInstances() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return instances_;
    }

    uint32_t instanceCount() {
        return instances_.size();
    }

    void* getInstanceByID(InstanceID id) {
        for (const auto& instance : instances_) {
            if (instance.id == id)
                return instance.ptr;
        }
        return nullptr;
    }

    const std::string getInstanceName(InstanceID id) {
        for (const auto& instance : instances_) {
            if (instance.id == id)
                return instance.name;
        }
        return nullptr;
    }

private:
    InstanceRegistry() = default;

    InstanceRegistry(const InstanceRegistry&) = delete;
    InstanceRegistry& operator=(const InstanceRegistry&) = delete;

    mutable std::mutex mutex_;

    std::vector<Instance> instances_;
    InstanceID nextId_ = 1;
};


