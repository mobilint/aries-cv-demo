#include <cstdlib>
#include <iostream>
#include <string>

#ifndef _WIN32
#include <termios.h>
#include <unistd.h>
#endif

#include "demo/demo_catalog.h"
#include "demo/demo_runtime.h"

namespace {
enum class LauncherKey { Up, Down, Enter, Quit, Unknown };

void printUsage(const DemoRegistry& registry, const char* argv0) {
    std::cout << "Usage: " << argv0 << " [demo-id] [--debug]\n";
    std::cout << "       " << argv0 << " --list\n\n";
    std::cout << "Options:\n";
    std::cout << "  --debug, -d  Print per-stage inference benchmarks to console\n\n";
    std::cout << "Available demos:\n";
    for (const auto& definition : registry.list()) {
        std::cout << "  " << definition.id() << "  -  " << definition.title() << "\n";
    }
}

#ifndef _WIN32
class TerminalRawMode {
public:
    TerminalRawMode() {
        if (!isatty(STDIN_FILENO)) return;
        if (tcgetattr(STDIN_FILENO, &mOriginal) != 0) return;
        termios raw = mOriginal;
        raw.c_lflag &= static_cast<unsigned long>(~(ICANON | ECHO));
        raw.c_cc[VMIN] = 1;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0) {
            mEnabled = true;
        }
    }

    ~TerminalRawMode() {
        if (mEnabled) {
            tcsetattr(STDIN_FILENO, TCSANOW, &mOriginal);
        }
    }

    bool enabled() const { return mEnabled; }

private:
    termios mOriginal{};
    bool mEnabled = false;
};

LauncherKey readLauncherKey() {
    char ch = 0;
    if (read(STDIN_FILENO, &ch, 1) != 1) return LauncherKey::Unknown;

    if (ch == '\n' || ch == '\r') return LauncherKey::Enter;
    if (ch == 'q' || ch == 'Q' || ch == 27) {
        if (ch != 27) return LauncherKey::Quit;

        char seq[2] = {0, 0};
        const ssize_t n = read(STDIN_FILENO, seq, 2);
        if (n == 2 && seq[0] == '[') {
            if (seq[1] == 'A') return LauncherKey::Up;
            if (seq[1] == 'B') return LauncherKey::Down;
        }
        return LauncherKey::Quit;
    }
    if (ch == 'k' || ch == 'K') return LauncherKey::Up;
    if (ch == 'j' || ch == 'J') return LauncherKey::Down;
    return LauncherKey::Unknown;
}
#else
LauncherKey readLauncherKey() { return LauncherKey::Enter; }
#endif

void renderLauncher(const DemoRegistry& registry, size_t selected, int accelerator_count) {
    std::cout << "\033[2J\033[H";
    std::cout << "Mobilint CV Demo\n";
    std::cout << "Detected accelerators: " << accelerator_count << "\n";
    std::cout << "Use Up/Down to select, Enter to run, q to quit.\n\n";
    for (size_t i = 0; i < registry.list().size(); ++i) {
        const auto& definition = registry.list()[i];
        std::cout << (i == selected ? "> " : "  ");
        std::cout << definition.title();
        std::cout << "\n";
    }
    std::cout.flush();
}

const DemoDefinition* runLauncher(const DemoRegistry& registry, int accelerator_count) {
    if (registry.list().empty()) return nullptr;

#ifndef _WIN32
    TerminalRawMode raw_mode;
    if (!raw_mode.enabled()) {
        return &registry.loadDefault();
    }
#endif

    size_t selected = 0;
    while (true) {
        renderLauncher(registry, selected, accelerator_count);
        switch (readLauncherKey()) {
            case LauncherKey::Up:
                selected = selected == 0 ? registry.list().size() - 1 : selected - 1;
                break;
            case LauncherKey::Down:
                selected = (selected + 1) % registry.list().size();
                break;
            case LauncherKey::Enter:
                std::cout << "\033[2J\033[H";
                return &registry.list()[selected];
            case LauncherKey::Quit:
                std::cout << "\033[2J\033[H";
                return nullptr;
            case LauncherKey::Unknown:
                break;
        }
    }
}
}  // namespace

int main(int argc, char* argv[]) {
    const int accelerator_count = detectAvailableAcceleratorCount();
    DemoRegistry registry;
    if (registry.list().empty()) {
        std::cerr << "No demos found under assets/*/config/demo.yaml\n";
        return 1;
    }

    bool debug_mode = false;
    std::string demo_id;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--debug" || arg == "-d") {
            debug_mode = true;
        } else if (arg == "--list" || arg == "-l" || arg == "--help" || arg == "-h") {
            printUsage(registry, argv[0]);
            return 0;
        } else {
            demo_id = arg;
        }
    }

    const DemoDefinition* definition = nullptr;
    if (!demo_id.empty()) {
        definition = registry.find(demo_id);
        if (!definition) {
            std::cerr << "Unknown demo id: " << demo_id << "\n\n";
            printUsage(registry, argv[0]);
            return 1;
        }
    } else {
        definition = runLauncher(registry, accelerator_count);
        if (!definition) return 0;
    }

    DemoRuntime runtime(*definition);
    if (debug_mode) runtime.setDebugMode(true);
    runtime.run();
    return 0;
}
