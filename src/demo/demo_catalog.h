#ifndef DEMO_INCLUDE_DEMO_CATALOG_H_
#define DEMO_INCLUDE_DEMO_CATALOG_H_

#include <memory>
#include <string>
#include <vector>

#include "demo/define.h"

class ModelPipeline;
class OverlayRenderer;

DemoManifest loadDemoManifest(const std::string& manifest_path, const std::string& mode = "");
int detectAvailableAcceleratorCount();

class DemoDefinition {
public:
    DemoDefinition(std::string manifest_path);

    const std::string& id() const { return mId; }
    const std::string& title() const { return mTitle; }
    const std::string& manifest_path() const { return mManifestPath; }

    DemoManifest loadManifest(const std::string& mode = "") const;
    std::unique_ptr<OverlayRenderer> createOverlayRenderer(const DemoManifest& manifest) const;
    std::vector<std::unique_ptr<ModelPipeline>> createPipelines(
        const DemoManifest& manifest) const;

private:
    std::string mId;
    std::string mTitle;
    std::string mManifestPath;
};

class DemoRegistry {
public:
    DemoRegistry();

    const std::vector<DemoDefinition>& list() const { return mDefinitions; }
    const DemoDefinition* find(const std::string& id) const;
    const DemoDefinition& loadDefault() const;

private:
    std::vector<DemoDefinition> mDefinitions;
};

#endif
