#include "demo/demo_catalog.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>

#include "demo/model_pipeline.h"
#include "demo/overlay.h"
#include "qbruntime/qbruntime.h"
#include "yaml-cpp/yaml.h"

namespace fs = std::filesystem;

namespace {
std::string findManifestPath(const std::string& relative_path) {
    const std::vector<fs::path> candidates = {
        fs::path(relative_path),
        fs::path("..") / relative_path,
        fs::path("../..") / relative_path,
    };
    for (const auto& candidate : candidates) {
        if (fs::exists(candidate)) {
            return candidate.lexically_normal().string();
        }
    }
    return relative_path;
}

std::string findAssetsRoot() {
    return findManifestPath("assets");
}

void loadDemoDefinitionHeader(const std::string& manifest_path, std::string& id, std::string& title) {
    YAML::Node root = YAML::LoadFile(manifest_path);
    id = root["id"].as<std::string>();
    title = root["title"].as<std::string>();
}

std::string resolvePath(const std::string& base_dir, const std::string& path) {
    if (path.empty()) return path;
    fs::path as_path(path);
    if (as_path.is_absolute()) return as_path.lexically_normal().string();
    fs::path resolved = fs::path(base_dir) / path;
    return resolved.lexically_normal().string();
}

std::string resolveSourcePath(FeederType feeder_type, const std::string& base_dir,
                              const std::string& source) {
    if (feeder_type == FeederType::VIDEO) {
        return resolvePath(base_dir, source);
    }
    return source;
}

DemoModeSetting loadModeSetting(const YAML::Node& node) {
    DemoModeSetting mode;
    mode.layout_setting = node["layout_setting"].as<std::string>();
    mode.feeder_setting = node["feeder_setting"].as<std::string>();
    mode.model_setting = node["model_setting"].as<std::string>();
    return mode;
}

bool canCreateAccelerator(int dev_no) {
    mobilint::StatusCode sc;
    auto acc = mobilint::Accelerator::create(dev_no, sc);
    return sc == mobilint::StatusCode::OK;
}

int detectAvailableAcceleratorCountImpl() {
    static int cached_count = -1;
    if (cached_count >= 0) return cached_count;

    cached_count = 0;
    for (int dev_no = 0; dev_no < 4; ++dev_no) {
        if (!canCreateAccelerator(dev_no)) break;
        cached_count++;
    }
    return cached_count;
}

std::string selectAutoMode(const std::map<std::string, DemoModeSetting>& modes) {
    const int acc_count = detectAvailableAcceleratorCountImpl();

    if (modes.find("mla400") != modes.end() && acc_count >= 4) {
        return "mla400";
    }
    if (modes.find("mla100") != modes.end() && acc_count >= 1) {
        return "mla100";
    }
    throw std::invalid_argument("No compatible demo mode for available accelerators.");
}

FeederType parseFeederType(const std::string& value) {
    static const std::map<std::string, FeederType> kMap = {
        {"CAMERA", FeederType::CAMERA},
        {"VIDEO", FeederType::VIDEO},
        {"IPCAMERA", FeederType::IPCAMERA},
        {"YOUTUBE", FeederType::YOUTUBE},
    };
    return kMap.at(value);
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

ModelType parseModelType(const std::string& value) {
    const std::string key = toLower(value);
    static const std::map<std::string, ModelType> kMap = {
        {"ssd", ModelType::SSD},
        {"face", ModelType::FACE},
        {"pose", ModelType::POSE},
        {"object", ModelType::OBJECT},
        {"segmentation", ModelType::SEGMENTATION},
        {"classification", ModelType::CLASSIFICATION},
        {"cls", ModelType::CLASSIFICATION},
    };
    auto it = kMap.find(key);
    if (it == kMap.end()) throw std::invalid_argument("Unsupported model_type: " + value);
    return it->second;
}

PostProcessType parsePostProcessType(const std::string& value) {
    const std::string key = toLower(value);
    static const std::map<std::string, PostProcessType> kMap = {
        {"", PostProcessType::NONE},
        {"none", PostProcessType::NONE},
        {"anchor", PostProcessType::ANCHOR},
        {"anchorless", PostProcessType::ANCHORLESS},
        {"dflfree", PostProcessType::DFLFREE},
        {"nmsfree", PostProcessType::NMSFREE},
        {"yolo11", PostProcessType::YOLO11},
        {"yolo26_inline", PostProcessType::YOLO26_INLINE},
        {"classification", PostProcessType::CLASSIFICATION},
        {"cls", PostProcessType::CLASSIFICATION},
    };
    auto it = kMap.find(key);
    if (it == kMap.end()) throw std::invalid_argument("Unsupported post type: " + value);
    return it->second;
}

InputDataType parseInputType(const std::string& value) {
    const std::string key = toLower(value);
    if (key == "uint8") return InputDataType::UINT8;
    if (key == "float32") return InputDataType::FLOAT32;
    throw std::invalid_argument("Unsupported input_type: " + value);
}

mobilint::Cluster parseCluster(const std::string& value) {
    if (value == "Cluster0") return mobilint::Cluster::Cluster0;
    if (value == "Cluster1") return mobilint::Cluster::Cluster1;
    throw std::invalid_argument("Unsupported cluster: " + value);
}

mobilint::Core parseCore(const std::string& value) {
    if (value == "Core0") return mobilint::Core::Core0;
    if (value == "Core1") return mobilint::Core::Core1;
    if (value == "Core2") return mobilint::Core::Core2;
    if (value == "Core3") return mobilint::Core::Core3;
    throw std::invalid_argument("Unsupported core: " + value);
}

void loadCoreIds(const YAML::Node& node, ModelSetting& model) {
    YAML::Node core_ids = node["core_id"];
    if (!core_ids || core_ids.size() == 0) return;

    model.use_core_id = true;
    model.core_id.clear();
    for (int j = 0; j < core_ids.size(); ++j) {
        const auto cluster = parseCluster(core_ids[j]["cluster"].as<std::string>());
        const auto core = parseCore(core_ids[j]["core"].as<std::string>());
        model.core_id.push_back({cluster, core});
    }
    model.num_core = static_cast<int>(model.core_id.size());
}

void applyLegacyPipelineMapping(const YAML::Node& node, ModelSetting& model) {
    const std::string pipeline_type = node["pipeline_type"].as<std::string>();
    model.model_type = ModelType::OBJECT;
    if (pipeline_type == "yolo11_det") {
        model.post_type = PostProcessType::YOLO11;
    } else if (pipeline_type == "yolo26_det") {
        model.post_type = PostProcessType::YOLO26_INLINE;
    } else if (pipeline_type == "yolo_anchorless_det") {
        model.post_type = PostProcessType::ANCHORLESS;
    } else {
        throw std::invalid_argument("Unsupported legacy pipeline_type: " + pipeline_type);
    }
}

void loadLayoutSetting(const std::string& path, LayoutSetting& layout_setting) {
    YAML::Node layout = YAML::LoadFile(path);
    layout_setting.canvas_size = cv::Size(layout["canvas_size"][0].as<int>(),
                                          layout["canvas_size"][1].as<int>());
    layout_setting.preview_asset =
        resolvePath(fs::path(path).parent_path().string(), layout["preview_asset"].as<std::string>());

    YAML::Node splash_assets = layout["splash_assets"];
    if (splash_assets) {
        for (int i = 0; i < splash_assets.size(); ++i) {
            layout_setting.splash_assets.push_back(resolvePath(
                fs::path(path).parent_path().string(), splash_assets[i].as<std::string>()));
        }
    }

    YAML::Node background_images = layout["background_images"];
    for (int i = 0; i < background_images.size(); ++i) {
        BackgroundImageLayout image_layout;
        const std::string image_path = resolvePath(fs::path(path).parent_path().string(),
                                                   background_images[i]["path"].as<std::string>());
        const int x = background_images[i]["roi"][0].as<int>();
        const int y = background_images[i]["roi"][1].as<int>();
        const int w = background_images[i]["roi"][2].as<int>();
        const int h = background_images[i]["roi"][3].as<int>();
        image_layout.roi = cv::Rect(x, y, w, h);
        image_layout.img = cv::imread(image_path);
        cv::resize(image_layout.img, image_layout.img, cv::Size(w, h));
        layout_setting.background_images.push_back(image_layout);
    }

    YAML::Node worker_tiles = layout["worker_tiles"];
    for (int i = 0; i < worker_tiles.size(); ++i) {
        WorkerLayout worker;
        worker.feeder_index = worker_tiles[i]["feeder_index"].as<int>();
        worker.model_index = worker_tiles[i]["model_index"].as<int>();
        worker.roi = cv::Rect(worker_tiles[i]["roi"][0].as<int>(),
                              worker_tiles[i]["roi"][1].as<int>(),
                              worker_tiles[i]["roi"][2].as<int>(),
                              worker_tiles[i]["roi"][3].as<int>());
        layout_setting.worker_tiles.push_back(worker);
    }
}

void loadFeederSetting(const std::string& path, std::vector<FeederSetting>& feeders_out) {
    YAML::Node feeders = YAML::LoadFile(path);
    for (int i = 0; i < feeders.size(); ++i) {
        FeederSetting feeder;
        feeder.feeder_type = parseFeederType(feeders[i]["type"].as<std::string>());
        YAML::Node sources = feeders[i]["sources"];
        for (int j = 0; j < sources.size(); ++j) {
            feeder.sources.push_back(resolveSourcePath(
                feeder.feeder_type, fs::path(path).parent_path().string(),
                sources[j].as<std::string>()));
        }
        feeders_out.push_back(feeder);
    }
}

void applyPipelineConfig(const YAML::Node& pipeline_config, PipelineConfig& out) {
    if (!pipeline_config) return;
    if (pipeline_config["num_classes"]) {
        out.num_classes = pipeline_config["num_classes"].as<int>();
    }
    YAML::Node labels = pipeline_config["labels"];
    if (labels) {
        out.labels.clear();
        for (int i = 0; i < labels.size(); ++i) {
            out.labels.push_back(labels[i].as<std::string>());
        }
    }
    YAML::Node display_labels = pipeline_config["display_labels"];
    if (display_labels) {
        out.display_labels.clear();
        for (int i = 0; i < display_labels.size(); ++i) {
            out.display_labels.push_back(display_labels[i].as<std::string>());
        }
    }
    YAML::Node label_colors = pipeline_config["label_colors"];
    if (label_colors && label_colors.IsMap()) {
        out.label_colors.clear();
        for (const auto& kv : label_colors) {
            YAML::Node bgr = kv.second;
            if (!bgr || !bgr.IsSequence() || bgr.size() != 3) continue;
            out.label_colors[kv.first.as<std::string>()] = {
                bgr[0].as<int>(), bgr[1].as<int>(), bgr[2].as<int>()};
        }
    }
    if (pipeline_config["topk"]) {
        out.topk = pipeline_config["topk"].as<int>();
    }
    if (pipeline_config["conf_threshold"]) {
        out.conf_threshold = pipeline_config["conf_threshold"].as<float>();
    }
    if (pipeline_config["iou_threshold"]) {
        out.iou_threshold = pipeline_config["iou_threshold"].as<float>();
    }
    if (pipeline_config["bbox_thickness"]) {
        out.bbox_thickness = pipeline_config["bbox_thickness"].as<int>();
    }
    if (pipeline_config["decode_bbox"]) {
        out.decode_bbox = pipeline_config["decode_bbox"].as<bool>();
    }
    if (pipeline_config["draw_label_text"]) {
        out.draw_label_text = pipeline_config["draw_label_text"].as<bool>();
    }
    if (pipeline_config["draw_score_text"]) {
        out.draw_score_text = pipeline_config["draw_score_text"].as<bool>();
    }
    if (pipeline_config["draw_detection_border"]) {
        out.draw_detection_border = pipeline_config["draw_detection_border"].as<bool>();
    }
    if (pipeline_config["draw_mask"]) {
        out.draw_mask = pipeline_config["draw_mask"].as<bool>();
    }
}

void loadModelSetting(const std::string& path, std::vector<ModelSetting>& models_out) {
    YAML::Node root = YAML::LoadFile(path);
    if (!root.IsMap()) {
        throw std::invalid_argument("Model setting must be a map with models.");
    }
    YAML::Node model_nodes = root["models"];
    YAML::Node pipeline_defaults = root["pipeline_config_defaults"];

    for (int i = 0; i < model_nodes.size(); ++i) {
        ModelSetting model;
        applyPipelineConfig(pipeline_defaults, model.pipeline_config);

        const YAML::Node model_node = model_nodes[i];
        if (model_node["pipeline_type"]) {
            applyLegacyPipelineMapping(model_node, model);
            model.input_type = parseInputType(model_node["input_type"].as<std::string>());
        } else {
            model.model_type = parseModelType(model_node["model_type"].as<std::string>());
            YAML::Node subconfig = model_node["subconfig"];
            if (subconfig && subconfig["input"]) {
                model.input_type = parseInputType(subconfig["input"].as<std::string>());
            } else if (model_node["input_type"]) {
                model.input_type = parseInputType(model_node["input_type"].as<std::string>());
            } else if (model.model_type != ModelType::SSD) {
                throw std::invalid_argument("Non-SSD model must define subconfig.input: " + path);
            }

            if (subconfig && subconfig["post"]) {
                model.post_type = parsePostProcessType(subconfig["post"].as<std::string>());
            } else if (model.model_type == ModelType::SSD) {
                model.post_type = PostProcessType::NONE;
            } else {
                throw std::invalid_argument("Non-SSD model must define subconfig.post: " + path);
            }
        }
        model.mxq_path =
            resolvePath(fs::path(path).parent_path().string(), model_node["mxq_path"].as<std::string>());
        if (model_node["dev_no"]) {
            model.dev_no = model_node["dev_no"].as<int>();
        } else if (model_node["device"]) {
            model.dev_no = model_node["device"].as<int>();
        }
        if (model_node["num_core"]) {
            model.num_core = model_node["num_core"].as<int>();
        }

        loadCoreIds(model_node, model);

        if (model_node["worker_threads"]) {
            model.worker_threads = model_node["worker_threads"].as<int>();
        }

        applyPipelineConfig(model_node["pipeline_config"], model.pipeline_config);

        models_out.push_back(model);
    }
}
}  // namespace

DemoManifest loadDemoManifest(const std::string& manifest_path, const std::string& mode) {
    DemoManifest manifest;
    manifest.manifest_path = manifest_path;
    manifest.manifest_dir = fs::path(manifest_path).parent_path().string();

    YAML::Node root = YAML::LoadFile(manifest_path);
    manifest.id = root["id"].as<std::string>();
    manifest.title = root["title"].as<std::string>();
    const YAML::Node modes = root["modes"];
    const YAML::Node ui = root["ui"];
    if (!modes) {
        throw std::invalid_argument("Manifest must define modes.");
    }
    for (const auto& mode_node : modes) {
        const std::string mode_name = mode_node.first.as<std::string>();
        manifest.modes.emplace(mode_name, loadModeSetting(mode_node.second));
    }

    const std::string selected_mode = !mode.empty() ? mode : selectAutoMode(manifest.modes);
    manifest.active_mode = selected_mode;

    if (isDemoVerboseLogEnabled()) {
        std::cerr << "[CATALOG] load manifest"
                  << " path=" << manifest_path
                  << " demo_id=" << manifest.id
                  << " requested_mode=" << (mode.empty() ? "auto" : mode)
                  << " selected_mode=" << selected_mode
                  << " detected_accelerators=" << detectAvailableAcceleratorCountImpl()
                  << std::endl;
    }

    auto it = manifest.modes.find(selected_mode);
    if (it == manifest.modes.end()) {
        throw std::invalid_argument("Unknown mode in manifest: " + selected_mode);
    }
    const DemoModeSetting& selected_files = it->second;

    const std::string layout_path = resolvePath(manifest.manifest_dir, selected_files.layout_setting);
    const std::string feeder_path = resolvePath(manifest.manifest_dir, selected_files.feeder_setting);
    const std::string model_path = resolvePath(manifest.manifest_dir, selected_files.model_setting);

    if (isDemoVerboseLogEnabled()) {
        std::cerr << "[CATALOG] config paths"
                  << " layout=" << layout_path
                  << " feeder=" << feeder_path
                  << " model=" << model_path << std::endl;
    }

    loadLayoutSetting(layout_path, manifest.layout);
    loadFeederSetting(feeder_path,
                      manifest.feeders);
    loadModelSetting(model_path, manifest.models);
    if (isDemoVerboseLogEnabled()) {
        std::cerr << "[CATALOG] loaded counts"
                  << " feeders=" << manifest.feeders.size()
                  << " workers=" << manifest.layout.worker_tiles.size()
                  << " models=" << manifest.models.size() << std::endl;
    }
    manifest.ui.overlay_style = ui["overlay_style"].as<std::string>();

    return manifest;
}

DemoDefinition::DemoDefinition(std::string manifest_path)
    : mManifestPath(std::move(manifest_path)) {
    loadDemoDefinitionHeader(mManifestPath, mId, mTitle);
}

DemoManifest DemoDefinition::loadManifest(const std::string& mode) const {
    return loadDemoManifest(mManifestPath, mode);
}

std::unique_ptr<OverlayRenderer> DemoDefinition::createOverlayRenderer(
    const DemoManifest& manifest) const {
    return ::createOverlayRenderer(manifest.ui.overlay_style);
}

int detectAvailableAcceleratorCount() { return detectAvailableAcceleratorCountImpl(); }

std::vector<std::unique_ptr<ModelPipeline>> DemoDefinition::createPipelines(
    const DemoManifest& manifest) const {
    std::vector<std::unique_ptr<ModelPipeline>> pipelines;
    pipelines.reserve(manifest.models.size());
    for (const auto& model : manifest.models) {
        pipelines.push_back(createModelPipeline(model));
    }
    return pipelines;
}

DemoRegistry::DemoRegistry() {
    const std::string assets_root = findAssetsRoot();
    if (!fs::exists(assets_root)) return;

    std::vector<std::string> manifest_paths;
    for (const auto& entry : fs::directory_iterator(assets_root)) {
        if (!entry.is_directory()) continue;
        fs::path manifest_path = entry.path() / "config" / "demo.yaml";
        if (fs::exists(manifest_path)) {
            manifest_paths.push_back(manifest_path.lexically_normal().string());
        }
    }
    std::sort(manifest_paths.begin(), manifest_paths.end());
    for (const auto& manifest_path : manifest_paths) {
        mDefinitions.emplace_back(manifest_path);
    }
}

const DemoDefinition* DemoRegistry::find(const std::string& id) const {
    for (const auto& definition : mDefinitions) {
        if (definition.id() == id) return &definition;
    }
    return nullptr;
}

const DemoDefinition& DemoRegistry::loadDefault() const { return mDefinitions.front(); }
