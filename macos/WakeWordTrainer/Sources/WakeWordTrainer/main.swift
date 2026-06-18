import AppKit
import CryptoKit
import Darwin
import Foundation
@preconcurrency import WebKit

private let trainerPort = 8789
private let appDisplayName = "WakeWord Trainer"
private let appExecutableName = "WakeWordTrainer"
private let appBundleName = "WakeWord Trainer.app"

private enum BackendState: Equatable {
    case stopped
    case bootstrapping
    case starting
    case running
    case failed(String)

    var label: String {
        switch self {
        case .stopped:
            return "Stopped"
        case .bootstrapping:
            return "Setting up"
        case .starting:
            return "Starting"
        case .running:
            return "Running"
        case .failed(let message):
            return "Failed: \(message)"
        }
    }
}

private final class BackendManager {
    let supportRoot: URL
    let appRoot: URL
    let sourceRoot: URL
    let venvDir: URL
    let pythonRoot: URL
    let managedPythonDir: URL
    let logsDir: URL
    let webURL = URL(string: "http://127.0.0.1:\(trainerPort)")!

    var onStateChange: ((BackendState) -> Void)?
    var onLogAppend: ((String) -> Void)?

    private let usesBundledSource: Bool
    private var process: Process?
    private var logHandle: FileHandle?
    private var outputPipe: Pipe?
    private var selectedPythonPath: String?

    private(set) var state: BackendState = .stopped {
        didSet {
            DispatchQueue.main.async { [state, onStateChange] in
                onStateChange?(state)
            }
        }
    }

    init() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        supportRoot = home.appendingPathComponent(".taterwakewordtrainer", isDirectory: true)
        appRoot = supportRoot.appendingPathComponent("app", isDirectory: true)
        venvDir = supportRoot.appendingPathComponent("recorder-venv", isDirectory: true)
        pythonRoot = supportRoot.appendingPathComponent("python", isDirectory: true)
        managedPythonDir = pythonRoot.appendingPathComponent("cpython-3.11", isDirectory: true)
        logsDir = supportRoot.appendingPathComponent("logs", isDirectory: true)

        let environment = ProcessInfo.processInfo.environment
        if let raw = environment["WAKEWORD_TRAINER_SOURCE_DIR"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty {
            sourceRoot = URL(fileURLWithPath: NSString(string: raw).expandingTildeInPath, isDirectory: true)
            usesBundledSource = false
        } else {
            sourceRoot = appRoot.appendingPathComponent("current", isDirectory: true)
            usesBundledSource = true
        }
    }

    func start() {
        if isManagedProcessRunning() {
            appendLauncherLog("Start requested; backend process is already running.\n")
            state = .running
            return
        }

        if isWebReady() {
            appendLog("WakeWord Trainer is already reachable at \(webURL.absoluteString)\n")
            appendLauncherLog("Start requested; web UI is already ready.\n")
            state = .running
            return
        }

        appendLauncherLog("Start requested.\n")
        state = .bootstrapping
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                self.appendLauncherLog("Ensuring private folders.\n")
                try self.ensurePrivateFolders()
                self.appendLauncherLog("Preparing bundled trainer source.\n")
                try self.prepareSource()
                self.appendLauncherLog("Ensuring Python runtime.\n")
                try self.ensurePythonRuntime()
                self.appendLauncherLog("Launching trainer backend.\n")
                try self.launchBackend()
            } catch {
                self.appendLauncherLog("Start failed: \(error.localizedDescription)\n")
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    func restart() {
        stop()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.start()
        }
    }

    func stop(waitForExit: Bool = false) {
        guard let process else {
            state = .stopped
            return
        }

        if process.isRunning {
            process.terminate()
            if waitForExit {
                let deadline = Date().addingTimeInterval(12)
                while process.isRunning && Date() < deadline {
                    Thread.sleep(forTimeInterval: 0.1)
                }
                if process.isRunning {
                    Darwin.kill(process.processIdentifier, SIGKILL)
                }
                process.waitUntilExit()
            } else {
                DispatchQueue.global(qos: .utility).async {
                    process.waitUntilExit()
                }
            }
        }
        closeLogHandle()
        self.process = nil
        state = .stopped
    }

    func openLogsFolder() {
        NSWorkspace.shared.activateFileViewerSelecting([logsDir])
    }

    func recoverIfBackendMissing() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self else { return }
            if self.isWebReady() {
                if self.state != .running {
                    self.appendLauncherLog("Recovery found web UI ready; marking running.\n")
                    self.state = .running
                }
                return
            }
            guard self.process == nil else {
                return
            }
            switch self.state {
            case .bootstrapping, .starting:
                guard FileManager.default.fileExists(atPath: self.sourceRoot.appendingPathComponent("run.sh").path) else {
                    return
                }
                self.appendLauncherLog("Recovery detected source with no backend; launching backend.\n")
                do {
                    try self.launchBackend()
                } catch {
                    self.appendLauncherLog("Recovery launch failed: \(error.localizedDescription)\n")
                    self.state = .failed(error.localizedDescription)
                }
            case .stopped, .running, .failed:
                return
            }
        }
    }

    func recentLogText(maxBytes: Int = 180_000) -> String {
        let urls = [
            logsDir.appendingPathComponent("launcher.log"),
            logsDir.appendingPathComponent("trainer.log")
        ]
        var chunks: [String] = []
        for url in urls where FileManager.default.fileExists(atPath: url.path) {
            if let text = tailText(from: url, maxBytes: maxBytes / max(1, urls.count)), !text.isEmpty {
                chunks.append("== \(url.lastPathComponent) ==\n\(text)")
            }
        }
        return chunks.joined(separator: "\n\n")
    }

    private func tailText(from url: URL, maxBytes: Int) -> String? {
        guard
            let handle = try? FileHandle(forReadingFrom: url),
            let size = try? handle.seekToEnd()
        else {
            return nil
        }
        let readSize = UInt64(max(1, maxBytes))
        let start = size > readSize ? size - readSize : 0
        do {
            try handle.seek(toOffset: start)
            let data = try handle.readToEnd() ?? Data()
            try handle.close()
            return String(decoding: data, as: UTF8.self)
        } catch {
            try? handle.close()
            return nil
        }
    }

    private func ensurePrivateFolders() throws {
        let folders = [
            supportRoot,
            appRoot,
            sourceRoot,
            venvDir,
            pythonRoot,
            managedPythonDir,
            logsDir
        ]

        for folder in folders {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        }
    }

    private func prepareSource() throws {
        guard usesBundledSource else {
            appendLog("Using external trainer source: \(sourceRoot.path)\n")
            return
        }

        guard let bundledSource = Bundle.main.resourceURL?.appendingPathComponent("TrainerSource", isDirectory: true),
              FileManager.default.fileExists(atPath: bundledSource.appendingPathComponent("run.sh").path)
        else {
            throw LauncherError("Bundled trainer source is missing from the app resources.")
        }

        try FileManager.default.createDirectory(at: sourceRoot, withIntermediateDirectories: true)
        try runCheckedProcess(
            executable: "/usr/bin/rsync",
            arguments: sourceSyncArguments(from: bundledSource, to: sourceRoot),
            currentDirectory: nil
        )
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: sourceRoot.appendingPathComponent("run.sh").path
        )
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: sourceRoot.appendingPathComponent("train_microwakeword_macos.sh").path
        )
        appendLog("Trainer source ready at \(sourceRoot.path)\n")
    }

    private func sourceSyncArguments(from source: URL, to destination: URL) -> [String] {
        var arguments = ["-a", "--delete"]
        let excludes = [
            ".DS_Store",
            ".git/",
            ".github/",
            ".agents/",
            ".codex/",
            ".recorder-venv/",
            ".venv/",
            "__pycache__/",
            "*.pyc",
            "macos/",
            "personal_samples/",
            "negative_samples/",
            "captured_audio/",
            "trim_history/",
            "trained_wake_words/",
            "trained_models/",
            "generated_samples/",
            "generated_augmented_features/",
            "personal_augmented_features/",
            "reviewed_negative_features/",
            "micro-wake-word/",
            "piper-sample-generator/",
            "mit_rirs/",
            "audioset/",
            "audioset_16k/",
            "fma/",
            "fma_16k/",
            "wham_16k/",
            "chime_16k/"
        ]
        for value in excludes {
            arguments.append("--exclude=\(value)")
        }
        arguments.append(source.path + "/")
        arguments.append(destination.path + "/")
        return arguments
    }

    private func ensurePythonRuntime() throws {
        let managedPython = managedPythonDir.appendingPathComponent("bin/python3.11").path
        if isUsablePython(managedPython) {
            selectedPythonPath = managedPython
            appendLog("Using managed Python: \(managedPython)\n")
            return
        }

        for candidate in systemPythonCandidates() {
            if isUsablePython(candidate) {
                selectedPythonPath = candidate
                appendLog("Using local Python: \(candidate)\n")
                return
            }
        }

        try installManagedPython()
    }

    private func systemPythonCandidates() -> [String] {
        [
            "/opt/homebrew/opt/python@3.11/bin/python3.11",
            "/opt/homebrew/bin/python3.11",
            "/usr/local/opt/python@3.11/bin/python3.11",
            "/usr/local/bin/python3.11",
            "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11"
        ]
    }

    private func isUsablePython(_ path: String) -> Bool {
        guard FileManager.default.isExecutableFile(atPath: path) else {
            return false
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = ["-c", "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }

    private func installManagedPython() throws {
        appendLog("No local Python 3.11 found. Installing managed Python under \(pythonRoot.path)\n")
        try FileManager.default.createDirectory(at: pythonRoot, withIntermediateDirectories: true)

        let assetURL = try findStandalonePythonAssetURL()
        appendLog("Downloading \(assetURL.absoluteString)\n")

        let archiveURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("wakeword-python-\(UUID().uuidString).tar.gz")
        let extractDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("wakeword-python-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: archiveURL)
            try? FileManager.default.removeItem(at: extractDir)
        }

        try downloadFile(from: assetURL, to: archiveURL)
        try FileManager.default.createDirectory(at: extractDir, withIntermediateDirectories: true)
        try runCheckedProcess(
            executable: "/usr/bin/tar",
            arguments: ["-xzf", archiveURL.path, "-C", extractDir.path],
            currentDirectory: nil
        )

        let extractedRoot = try findExtractedPythonRoot(in: extractDir)
        if FileManager.default.fileExists(atPath: managedPythonDir.path) {
            try FileManager.default.removeItem(at: managedPythonDir)
        }
        try FileManager.default.moveItem(at: extractedRoot, to: managedPythonDir)

        let python = managedPythonDir.appendingPathComponent("bin/python3.11").path
        guard isUsablePython(python) else {
            throw LauncherError("Managed Python installed, but \(python) did not run as Python 3.11.")
        }
        selectedPythonPath = python
        appendLog("Managed Python ready: \(python)\n")
    }

    private func findStandalonePythonAssetURL() throws -> URL {
        let releasesURL = URL(string: "https://api.github.com/repos/astral-sh/python-build-standalone/releases?per_page=20")!
        var request = URLRequest(url: releasesURL)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue(appExecutableName, forHTTPHeaderField: "User-Agent")
        request.timeoutInterval = 60

        let data = try loadData(from: request)
        guard let releases = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw LauncherError("Could not read python-build-standalone releases.")
        }

        let targetArch = standalonePythonArch()
        for release in releases {
            guard let assets = release["assets"] as? [[String: Any]] else {
                continue
            }
            for asset in assets {
                guard
                    let name = asset["name"] as? String,
                    let rawURL = asset["browser_download_url"] as? String,
                    name.hasPrefix("cpython-3.11."),
                    name.contains("-\(targetArch)-apple-darwin-install_only.tar.gz"),
                    !name.contains("stripped"),
                    let url = URL(string: rawURL)
                else {
                    continue
                }
                return url
            }
        }

        throw LauncherError("Could not find a standalone Python 3.11 build for \(targetArch)-apple-darwin.")
    }

    private func standalonePythonArch() -> String {
        #if arch(arm64)
        return "aarch64"
        #else
        return "x86_64"
        #endif
    }

    private func findExtractedPythonRoot(in directory: URL) throws -> URL {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw LauncherError("Could not inspect extracted Python archive.")
        }

        for case let url as URL in enumerator {
            if url.lastPathComponent == "python3.11",
               url.deletingLastPathComponent().lastPathComponent == "bin" {
                return url.deletingLastPathComponent().deletingLastPathComponent()
            }
        }
        throw LauncherError("Downloaded Python archive did not contain bin/python3.11.")
    }

    private func launchBackend() throws {
        state = .starting

        guard FileManager.default.fileExists(atPath: sourceRoot.appendingPathComponent("run.sh").path) else {
            throw LauncherError("Could not find run.sh in \(sourceRoot.path)")
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = ["run.sh"]
        process.currentDirectoryURL = sourceRoot
        process.environment = backendEnvironment()

        appendLog("Starting WakeWord Trainer on \(webURL.absoluteString)\n")
        appendLauncherLog("Opening backend log and starting run.sh.\n")

        let handle = try openLog(named: "trainer.log", append: true)
        logHandle = handle
        outputPipe = streamProcessOutput(process, to: handle)
        process.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                guard let self else { return }
                self.outputPipe?.fileHandleForReading.readabilityHandler = nil
                self.outputPipe = nil
                self.closeLogHandle()
                self.process = nil
                if proc.terminationStatus == 0 {
                    self.state = .stopped
                } else {
                    self.state = .failed("Trainer exited with status \(proc.terminationStatus)")
                }
            }
        }

        try process.run()
        self.process = process
        appendLauncherLog("Backend process started with pid \(process.processIdentifier).\n")

        if waitForWebReady(timeout: 90) {
            appendLauncherLog("Trainer web UI is ready.\n")
            state = .running
        } else if process.isRunning {
            appendLauncherLog("Trainer is still starting after readiness timeout.\n")
            state = .starting
        } else {
            throw LauncherError("Trainer exited before the web UI became ready.")
        }
    }

    private func backendEnvironment() -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        let pathPrefix = [
            "/opt/homebrew/opt/python@3.11/bin",
            "/opt/homebrew/bin",
            "/usr/local/opt/python@3.11/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin"
        ]
        let existingPath = environment["PATH"] ?? ""

        environment["PATH"] = (pathPrefix + [existingPath]).filter { !$0.isEmpty }.joined(separator: ":")
        if let selectedPythonPath {
            environment["REC_PYTHON_BIN"] = selectedPythonPath
            environment["PYTHON_BIN"] = selectedPythonPath
        }
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["REC_VENV_DIR"] = venvDir.path
        environment["REC_HOST"] = "0.0.0.0"
        environment["REC_PORT"] = "\(trainerPort)"
        environment["WAKEWORD_TRAINER_SUPPORT_DIR"] = supportRoot.path
        return environment
    }

    private func openLog(named name: String, append: Bool) throws -> FileHandle {
        try FileManager.default.createDirectory(at: logsDir, withIntermediateDirectories: true)
        let url = logsDir.appendingPathComponent(name)
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: url)
        if append {
            try handle.seekToEnd()
            let header = "\n\n=== \(Date()) ===\n".data(using: .utf8) ?? Data()
            try handle.write(contentsOf: header)
        } else {
            try handle.truncate(atOffset: 0)
        }
        return handle
    }

    private func streamProcessOutput(_ process: Process, to handle: FileHandle) -> Pipe {
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] reader in
            let data = reader.availableData
            guard !data.isEmpty else { return }
            try? handle.write(contentsOf: data)
            if let text = String(data: data, encoding: .utf8) {
                self?.appendLog(text)
            } else {
                self?.appendLog(String(decoding: data, as: UTF8.self))
            }
        }
        return pipe
    }

    private func appendLog(_ text: String) {
        guard !text.isEmpty else { return }
        DispatchQueue.main.async { [onLogAppend] in
            onLogAppend?(text)
        }
    }

    private func appendLauncherLog(_ text: String) {
        guard !text.isEmpty else { return }
        do {
            try FileManager.default.createDirectory(at: logsDir, withIntermediateDirectories: true)
            let url = logsDir.appendingPathComponent("launcher.log")
            if !FileManager.default.fileExists(atPath: url.path) {
                FileManager.default.createFile(atPath: url.path, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: url)
            try handle.seekToEnd()
            let stamp = ISO8601DateFormatter().string(from: Date())
            let data = "[\(stamp)] \(text)".data(using: .utf8) ?? Data()
            try handle.write(contentsOf: data)
            try handle.close()
        } catch {
            appendLog("Launcher log write failed: \(error.localizedDescription)\n")
        }
    }

    private func closeLogHandle() {
        try? logHandle?.close()
        logHandle = nil
    }

    private func isManagedProcessRunning() -> Bool {
        guard let process else { return false }
        return process.isRunning
    }

    private func waitForWebReady(timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if isWebReady() {
                return true
            }
            Thread.sleep(forTimeInterval: 0.75)
        }
        return false
    }

    private func isWebReady() -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
        process.arguments = [
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            "1.5",
            "--output",
            "/dev/null",
            webURL.absoluteString
        ]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }

    private func runCheckedProcess(executable: String, arguments: [String], currentDirectory: URL?) throws {
        appendLog("$ \(executable) \(arguments.joined(separator: " "))\n")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.currentDirectoryURL = currentDirectory

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] reader in
            let data = reader.availableData
            guard !data.isEmpty else { return }
            self?.appendLog(String(decoding: data, as: UTF8.self))
        }

        try process.run()
        process.waitUntilExit()
        pipe.fileHandleForReading.readabilityHandler = nil
        guard process.terminationStatus == 0 else {
            throw LauncherError("\(executable) exited with status \(process.terminationStatus).")
        }
    }

    private func loadData(from request: URLRequest) throws -> Data {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<Data, Error>?
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error {
                result = .failure(error)
            } else if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                result = .failure(LauncherError("Request failed with HTTP \(http.statusCode)."))
            } else {
                result = .success(data ?? Data())
            }
            semaphore.signal()
        }
        task.resume()
        semaphore.wait()

        guard let result else {
            throw LauncherError("Request did not complete.")
        }
        return try result.get()
    }

    private func downloadFile(from url: URL, to destination: URL) throws {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<URL, Error>?
        let task = URLSession.shared.downloadTask(with: url) { location, _, error in
            if let error {
                result = .failure(error)
            } else if let location {
                result = .success(location)
            } else {
                result = .failure(LauncherError("Download finished without a file."))
            }
            semaphore.signal()
        }
        task.resume()
        semaphore.wait()

        guard let result else {
            throw LauncherError("Download did not complete.")
        }
        let tempURL = try result.get()
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.copyItem(at: tempURL, to: destination)
    }
}

private struct UpdateManifest: Decodable, Equatable {
    let version: String
    let build: Int
    let url: URL
    let sha256: String
    let notes: String?
}

private enum UpdateState: Equatable {
    case idle
    case checking
    case current
    case available(UpdateManifest)
    case downloading(UpdateManifest)
    case installing(UpdateManifest)
    case failed(String)

    var isBusy: Bool {
        switch self {
        case .checking, .downloading, .installing:
            return true
        case .idle, .current, .available, .failed:
            return false
        }
    }
}

private final class UpdateManager {
    var onStateChange: ((UpdateState) -> Void)?

    private let updatesRoot: URL
    private var availableManifest: UpdateManifest?

    private(set) var state: UpdateState = .idle {
        didSet {
            DispatchQueue.main.async { [state, onStateChange] in
                onStateChange?(state)
            }
        }
    }

    init() {
        updatesRoot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".taterwakewordtrainer", isDirectory: true)
            .appendingPathComponent("updates", isDirectory: true)
    }

    func checkForUpdates(manual: Bool) {
        guard !state.isBusy else { return }
        state = .checking

        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self else { return }
            do {
                let manifest = try self.fetchManifest()
                if self.isNewerThanCurrent(manifest) {
                    self.availableManifest = manifest
                    self.state = .available(manifest)
                } else {
                    self.availableManifest = nil
                    self.state = manual ? .current : .idle
                }
            } catch {
                self.state = manual ? .failed(error.localizedDescription) : .idle
            }
        }
    }

    func installAvailableUpdate() {
        let manifest: UpdateManifest?
        switch state {
        case .available(let current):
            manifest = current
        case .failed, .current, .idle, .checking, .downloading, .installing:
            manifest = availableManifest
        }

        guard let manifest else { return }
        state = .downloading(manifest)

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                let newApp = try self.prepareUpdate(manifest)
                self.state = .installing(manifest)
                try self.launchInstaller(newApp: newApp)
                DispatchQueue.main.async {
                    NSApp.terminate(nil)
                }
            } catch {
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    private func fetchManifest() throws -> UpdateManifest {
        guard let url = manifestURL() else {
            throw LauncherError("No update manifest URL is configured.")
        }

        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        request.timeoutInterval = 30
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(appExecutableName, forHTTPHeaderField: "User-Agent")

        let data = try loadData(from: request)
        return try JSONDecoder().decode(UpdateManifest.self, from: data)
    }

    private func manifestURL() -> URL? {
        let environment = ProcessInfo.processInfo.environment
        for key in ["WAKEWORD_TRAINER_UPDATE_MANIFEST_URL", "MICROWAKEWORD_TRAINER_UPDATE_MANIFEST_URL"] {
            if let raw = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
               !raw.isEmpty,
               let url = URL(string: raw) {
                return url
            }
        }

        if let raw = Bundle.main.object(forInfoDictionaryKey: "WakeWordTrainerUpdateManifestURL") as? String {
            return URL(string: raw.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return nil
    }

    private func isNewerThanCurrent(_ manifest: UpdateManifest) -> Bool {
        let currentVersion = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
        let versionComparison = compareVersion(manifest.version, to: currentVersion)
        if versionComparison != .orderedSame {
            return versionComparison == .orderedDescending
        }

        let currentBuild = buildNumber(from: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String)
        return manifest.build > currentBuild
    }

    private func compareVersion(_ lhs: String, to rhs: String) -> ComparisonResult {
        let left = versionParts(lhs)
        let right = versionParts(rhs)
        for index in 0..<max(left.count, right.count) {
            let leftPart = index < left.count ? left[index] : 0
            let rightPart = index < right.count ? right[index] : 0
            if leftPart > rightPart {
                return .orderedDescending
            }
            if leftPart < rightPart {
                return .orderedAscending
            }
        }
        return .orderedSame
    }

    private func versionParts(_ raw: String) -> [Int] {
        raw.split { !$0.isNumber }.compactMap { Int($0) }
    }

    private func buildNumber(from raw: String?) -> Int {
        let value = raw ?? "0"
        return versionParts(value).first ?? 0
    }

    private func prepareUpdate(_ manifest: UpdateManifest) throws -> URL {
        try FileManager.default.createDirectory(at: updatesRoot, withIntermediateDirectories: true)

        let archiveName = "WakeWordTrainer-\(safePathComponent(versionLabel(manifest.version))).zip"
        let archiveURL = updatesRoot.appendingPathComponent(archiveName)
        let extractDir = updatesRoot.appendingPathComponent("staging-\(UUID().uuidString)", isDirectory: true)

        try downloadFile(from: manifest.url, to: archiveURL)
        try verifySHA256(of: archiveURL, expected: manifest.sha256)
        try FileManager.default.createDirectory(at: extractDir, withIntermediateDirectories: true)
        try runCheckedProcess(
            executable: "/usr/bin/ditto",
            arguments: ["-x", "-k", archiveURL.path, extractDir.path]
        )

        let newApp = try findExtractedApp(in: extractDir)
        guard FileManager.default.fileExists(
            atPath: newApp.appendingPathComponent("Contents/MacOS/\(appExecutableName)").path
        ) else {
            throw LauncherError("Downloaded update did not contain the WakeWord Trainer executable.")
        }
        return newApp
    }

    private func verifySHA256(of url: URL, expected rawExpected: String) throws {
        let expected = rawExpected.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let hex = CharacterSet(charactersIn: "0123456789abcdef")
        guard expected.count == 64, expected.unicodeScalars.allSatisfy({ hex.contains($0) }) else {
            throw LauncherError("Update manifest is missing a valid SHA-256 hash.")
        }

        let actual = try sha256Hex(of: url)
        guard actual == expected else {
            throw LauncherError("Downloaded update did not match the manifest SHA-256.")
        }
    }

    private func sha256Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            try? handle.close()
        }

        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1024 * 1024) ?? Data()
            if chunk.isEmpty {
                break
            }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func findExtractedApp(in directory: URL) throws -> URL {
        let preferred = directory.appendingPathComponent(appBundleName, isDirectory: true)
        if FileManager.default.fileExists(atPath: preferred.path) {
            return preferred
        }

        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw LauncherError("Could not inspect extracted update.")
        }

        for case let url as URL in enumerator where url.pathExtension == "app" {
            return url
        }
        throw LauncherError("Downloaded update did not contain a macOS app.")
    }

    private func launchInstaller(newApp: URL) throws {
        let targetApp = try currentAppURL()
        let scriptURL = try writeInstallerScript()

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = [scriptURL.path, "\(getpid())", newApp.path, targetApp.path]
        try process.run()
    }

    private func currentAppURL() throws -> URL {
        let bundleURL = Bundle.main.bundleURL.standardizedFileURL
        guard bundleURL.pathExtension == "app" else {
            throw LauncherError("WakeWord Trainer is not running from an app bundle.")
        }
        return bundleURL
    }

    private func writeInstallerScript() throws -> URL {
        try FileManager.default.createDirectory(at: updatesRoot, withIntermediateDirectories: true)
        let scriptURL = updatesRoot.appendingPathComponent("install-update-\(UUID().uuidString).sh")
        let script = """
        #!/bin/sh
        set -eu

        APP_PID="$1"
        NEW_APP="$2"
        TARGET_APP="$3"
        SCRIPT_PATH="$0"
        WAIT_COUNT=0

        while kill -0 "$APP_PID" 2>/dev/null && [ "$WAIT_COUNT" -lt 150 ]; do
          sleep 0.2
          WAIT_COUNT=$((WAIT_COUNT + 1))
        done

        TARGET_PARENT="$(dirname "$TARGET_APP")"
        TARGET_NAME="$(basename "$TARGET_APP")"
        STAGED="${TARGET_PARENT}/.${TARGET_NAME}.updating"
        BACKUP="${TARGET_PARENT}/.${TARGET_NAME}.previous"
        NEW_PARENT="$(dirname "$NEW_APP")"

        rm -rf "$STAGED"
        ditto "$NEW_APP" "$STAGED"
        rm -rf "$BACKUP"
        if [ -d "$TARGET_APP" ]; then
          mv "$TARGET_APP" "$BACKUP"
        fi
        mv "$STAGED" "$TARGET_APP"
        xattr -dr com.apple.quarantine "$TARGET_APP" 2>/dev/null || true
        open "$TARGET_APP"
        rm -rf "$NEW_PARENT"
        rm -f "$SCRIPT_PATH"
        """

        try script.write(to: scriptURL, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: scriptURL.path)
        return scriptURL
    }

    private func safePathComponent(_ raw: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        let value = raw.unicodeScalars.map { allowed.contains($0) ? String($0) : "-" }.joined()
        return value.isEmpty ? "update" : value
    }

    private func versionLabel(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.lowercased().hasPrefix("v") {
            return trimmed
        }
        return "v\(trimmed)"
    }

    private func loadData(from request: URLRequest) throws -> Data {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<Data, Error>?
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error {
                result = .failure(error)
            } else if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                result = .failure(LauncherError("Update request failed with HTTP \(http.statusCode)."))
            } else {
                result = .success(data ?? Data())
            }
            semaphore.signal()
        }
        task.resume()
        semaphore.wait()

        guard let result else {
            throw LauncherError("Update request did not complete.")
        }
        return try result.get()
    }

    private func downloadFile(from url: URL, to destination: URL) throws {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<URL, Error>?
        let task = URLSession.shared.downloadTask(with: url) { location, response, error in
            if let error {
                result = .failure(error)
            } else if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                result = .failure(LauncherError("Update download failed with HTTP \(http.statusCode)."))
            } else if let location {
                result = .success(location)
            } else {
                result = .failure(LauncherError("Update download finished without a file."))
            }
            semaphore.signal()
        }
        task.resume()
        semaphore.wait()

        guard let result else {
            throw LauncherError("Update download did not complete.")
        }

        let tempURL = try result.get()
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.copyItem(at: tempURL, to: destination)
    }

    private func runCheckedProcess(executable: String, arguments: [String]) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw LauncherError("\(executable) exited with status \(process.terminationStatus).")
        }
    }
}

private final class TrainerWindowController: NSWindowController, WKNavigationDelegate, WKUIDelegate {
    private let webView: WKWebView
    private let setupView: NSVisualEffectView
    private let setupLogoView: NSImageView
    private let progressIndicator: NSProgressIndicator
    private let statusTitleLabel: NSTextField
    private let statusDetailLabel: NSTextField
    private var outputBuffer = ""

    init() {
        webView = WKWebView(frame: .zero)
        setupView = NSVisualEffectView(frame: .zero)
        setupLogoView = NSImageView(frame: .zero)
        progressIndicator = NSProgressIndicator(frame: .zero)
        statusTitleLabel = NSTextField(labelWithString: "Starting WakeWord Trainer")
        statusDetailLabel = NSTextField(labelWithString: "Preparing the local capture server...")

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1420, height: 920),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = appDisplayName
        window.minSize = NSSize(width: 1050, height: 700)
        window.center()
        window.backgroundColor = .clear

        super.init(window: window)

        webView.navigationDelegate = self
        webView.uiDelegate = self

        let contentView = NSView()
        contentView.translatesAutoresizingMaskIntoConstraints = false
        webView.translatesAutoresizingMaskIntoConstraints = false
        setupView.translatesAutoresizingMaskIntoConstraints = false
        setupLogoView.translatesAutoresizingMaskIntoConstraints = false
        progressIndicator.translatesAutoresizingMaskIntoConstraints = false
        statusTitleLabel.translatesAutoresizingMaskIntoConstraints = false
        statusDetailLabel.translatesAutoresizingMaskIntoConstraints = false

        setupView.material = .hudWindow
        setupView.blendingMode = .withinWindow
        setupView.state = .active
        setupView.wantsLayer = true

        setupLogoView.image = bundledImage(named: "TaterRepoLogo", withExtension: "png")
        setupLogoView.imageScaling = .scaleProportionallyUpOrDown
        setupLogoView.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        setupLogoView.setContentCompressionResistancePriority(.defaultLow, for: .vertical)

        progressIndicator.style = .spinning
        progressIndicator.controlSize = .regular
        progressIndicator.startAnimation(nil)

        statusTitleLabel.font = NSFont.systemFont(ofSize: 22, weight: .semibold)
        statusTitleLabel.textColor = .labelColor
        statusTitleLabel.alignment = .center
        statusDetailLabel.font = NSFont.systemFont(ofSize: 14, weight: .regular)
        statusDetailLabel.textColor = .secondaryLabelColor
        statusDetailLabel.alignment = .center
        statusDetailLabel.lineBreakMode = .byTruncatingMiddle

        setupView.addSubview(setupLogoView)
        setupView.addSubview(progressIndicator)
        setupView.addSubview(statusTitleLabel)
        setupView.addSubview(statusDetailLabel)
        contentView.addSubview(webView)
        contentView.addSubview(setupView)
        window.contentView = contentView

        NSLayoutConstraint.activate([
            webView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            webView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            webView.topAnchor.constraint(equalTo: contentView.topAnchor),
            webView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
            setupView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            setupView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            setupView.topAnchor.constraint(equalTo: contentView.topAnchor),
            setupView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
            setupLogoView.centerXAnchor.constraint(equalTo: setupView.centerXAnchor),
            setupLogoView.centerYAnchor.constraint(equalTo: setupView.centerYAnchor, constant: -70),
            setupLogoView.widthAnchor.constraint(lessThanOrEqualTo: setupView.widthAnchor, multiplier: 0.68),
            setupLogoView.heightAnchor.constraint(lessThanOrEqualTo: setupView.heightAnchor, multiplier: 0.32),
            setupLogoView.widthAnchor.constraint(lessThanOrEqualToConstant: 760),
            progressIndicator.centerXAnchor.constraint(equalTo: setupView.centerXAnchor),
            progressIndicator.topAnchor.constraint(equalTo: setupLogoView.bottomAnchor, constant: 30),
            statusTitleLabel.leadingAnchor.constraint(equalTo: setupView.leadingAnchor, constant: 56),
            statusTitleLabel.trailingAnchor.constraint(equalTo: setupView.trailingAnchor, constant: -56),
            statusTitleLabel.topAnchor.constraint(equalTo: progressIndicator.bottomAnchor, constant: 20),
            statusDetailLabel.leadingAnchor.constraint(equalTo: setupView.leadingAnchor, constant: 72),
            statusDetailLabel.trailingAnchor.constraint(equalTo: setupView.trailingAnchor, constant: -72),
            statusDetailLabel.topAnchor.constraint(equalTo: statusTitleLabel.bottomAnchor, constant: 10)
        ])

        setStatus("Starting WakeWord Trainer", detail: "Preparing the local capture server...", webVisible: false)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func load(url: URL) {
        webView.load(URLRequest(url: url))
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        guard let url = navigationAction.request.url else {
            decisionHandler(.allow)
            return
        }
        if shouldOpenExternally(url) {
            NSWorkspace.shared.open(url)
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }

    func webView(
        _ webView: WKWebView,
        createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction,
        windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
        if let url = navigationAction.request.url {
            NSWorkspace.shared.open(url)
        }
        return nil
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptAlertPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping () -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = appDisplayName
        alert.informativeText = message
        alert.alertStyle = .informational
        alert.addButton(withTitle: "OK")

        if let window {
            alert.beginSheetModal(for: window) { _ in
                completionHandler()
            }
        } else {
            alert.runModal()
            completionHandler()
        }
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptConfirmPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping (Bool) -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = appDisplayName
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Continue")
        alert.addButton(withTitle: "Cancel")

        if let window {
            alert.beginSheetModal(for: window) { response in
                completionHandler(response == .alertFirstButtonReturn)
            }
        } else {
            completionHandler(alert.runModal() == .alertFirstButtonReturn)
        }
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptTextInputPanelWithPrompt prompt: String,
        defaultText: String?,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping (String?) -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = appDisplayName
        alert.informativeText = prompt
        alert.alertStyle = .informational
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")

        let input = NSTextField(string: defaultText ?? "")
        input.frame = NSRect(x: 0, y: 0, width: 320, height: 24)
        alert.accessoryView = input

        let finish: (NSApplication.ModalResponse) -> Void = { response in
            completionHandler(response == .alertFirstButtonReturn ? input.stringValue : nil)
        }

        if let window {
            alert.beginSheetModal(for: window, completionHandler: finish)
        } else {
            finish(alert.runModal())
        }
    }

    private func shouldOpenExternally(_ url: URL) -> Bool {
        guard let scheme = url.scheme?.lowercased() else {
            return false
        }
        if scheme != "http" && scheme != "https" {
            return !["about", "data", "blob"].contains(scheme)
        }
        let host = url.host?.lowercased() ?? ""
        let localHosts: Set<String> = ["127.0.0.1", "localhost", "::1"]
        if localHosts.contains(host), (url.port == trainerPort || url.port == nil) {
            return false
        }
        return true
    }

    func setStatus(_ title: String, detail: String = "", webVisible: Bool) {
        statusTitleLabel.stringValue = title
        if !detail.isEmpty {
            statusDetailLabel.stringValue = detail
        }
        setupView.isHidden = webVisible
        webView.isHidden = !webVisible
        if webVisible {
            progressIndicator.stopAnimation(nil)
        } else {
            progressIndicator.startAnimation(nil)
        }
    }

    func updateSetupProgress(from text: String) {
        guard !text.isEmpty else { return }
        outputBuffer += text
        let lines = outputBuffer.components(separatedBy: .newlines)
        outputBuffer = lines.last ?? ""
        for line in lines.dropLast() {
            if let status = setupStatus(from: line) {
                statusDetailLabel.stringValue = status
            }
        }
    }

    private func setupStatus(from rawLine: String) -> String? {
        let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty else { return nil }
        if line.contains("Trainer source ready") {
            return "Preparing the trainer files..."
        }
        if line.contains("Using local Python") || line.contains("Using managed Python") {
            return "Checking Python 3.11..."
        }
        if line.contains("Fresh trainer UI venv") {
            return "Installing the local web UI..."
        }
        if line.hasPrefix("Collecting ") {
            return "Installing \(packageName(from: line.dropFirst(11)))"
        }
        if line.hasPrefix("Using cached ") {
            return "Preparing cached package \(packageName(from: line.dropFirst(13)))"
        }
        if line.hasPrefix("Downloading ") {
            return "Downloading \(packageName(from: line.dropFirst(12)))"
        }
        if line.hasPrefix("Installing collected packages:") {
            return "Installing Python packages..."
        }
        if line.hasPrefix("Successfully installed") {
            return "Python packages installed"
        }
        if line.contains("Launching:") {
            return "Starting the local capture server..."
        }
        if line.contains("Uvicorn running on") {
            return "Opening WakeWord Trainer..."
        }
        return nil
    }

    private func packageName<S: StringProtocol>(from value: S) -> String {
        let token = value
            .split(separator: " ", maxSplits: 1)
            .first?
            .split(separator: "=", maxSplits: 1)
            .first?
            .trimmingCharacters(in: CharacterSet(charactersIn: ",;:()"))
        return token.flatMap { $0.isEmpty ? nil : String($0) } ?? "dependency"
    }

    private func bundledImage(named name: String, withExtension ext: String) -> NSImage? {
        guard let url = Bundle.main.url(forResource: name, withExtension: ext) else {
            return nil
        }
        return NSImage(contentsOf: url)
    }
}

private final class AppDelegate: NSObject, NSApplicationDelegate {
    private let backend = BackendManager()
    private let updater = UpdateManager()
    private var statusItem: NSStatusItem?
    private var statusMenuItem: NSMenuItem?
    private var startMenuItem: NSMenuItem?
    private var stopMenuItem: NSMenuItem?
    private var updateMenuItem: NSMenuItem?
    private var checkUpdatesMenuItem: NSMenuItem?
    private var windowController: TrainerWindowController?
    private var usesMenuBarImage = false
    private var recoveryTimer: Timer?
    private var updateMenuResetTimer: Timer?
    private var updateCheckTimer: Timer?
    private let automaticUpdateInterval: TimeInterval = 12 * 60 * 60
    private let lastAutomaticUpdateCheckKey = "LastAutomaticUpdateCheck"

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        configureStatusItem()

        backend.onStateChange = { [weak self] state in
            self?.refreshMenu(for: state)
            self?.refreshWindow(for: state)
        }
        backend.onLogAppend = { [weak self] text in
            self?.windowController?.updateSetupProgress(from: text)
        }
        updater.onStateChange = { [weak self] state in
            self?.refreshUpdateMenu(for: state)
        }

        showWindow()
        backend.start()
        startRecoveryWatchdog()
        startAutomaticUpdateChecks()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        recoveryTimer?.invalidate()
        updateCheckTimer?.invalidate()
        updateMenuResetTimer?.invalidate()
        backend.stop(waitForExit: true)
        return .terminateNow
    }

    private func startRecoveryWatchdog() {
        recoveryTimer?.invalidate()
        recoveryTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.backend.recoverIfBackendMissing()
        }
    }

    private func startAutomaticUpdateChecks() {
        updateCheckTimer?.invalidate()
        scheduleAutomaticUpdateCheck(after: automaticUpdateCheckDelay())
    }

    private func automaticUpdateCheckDelay() -> TimeInterval {
        let lastCheck = UserDefaults.standard.double(forKey: lastAutomaticUpdateCheckKey)
        guard lastCheck > 0 else {
            return 6
        }

        let elapsed = Date().timeIntervalSince1970 - lastCheck
        guard elapsed >= 0 else {
            return automaticUpdateInterval
        }

        if elapsed >= automaticUpdateInterval {
            return 6
        }
        return max(6, automaticUpdateInterval - elapsed)
    }

    private func scheduleAutomaticUpdateCheck(after delay: TimeInterval) {
        updateCheckTimer?.invalidate()
        updateCheckTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
            self?.runAutomaticUpdateCheck()
        }
    }

    private func runAutomaticUpdateCheck() {
        if updater.state.isBusy {
            scheduleAutomaticUpdateCheck(after: 60)
            return
        }

        UserDefaults.standard.set(Date().timeIntervalSince1970, forKey: lastAutomaticUpdateCheckKey)
        updater.checkForUpdates(manual: false)
        scheduleAutomaticUpdateCheck(after: automaticUpdateInterval)
    }

    private func configureStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let image = resourceImage(named: "WakeWordMenuBarTemplate", withExtension: "png") {
            image.isTemplate = true
            image.size = NSSize(width: 18, height: 18)
            item.button?.image = image
            item.button?.imagePosition = .imageOnly
            item.button?.title = ""
            usesMenuBarImage = true
        } else {
            item.button?.title = "W"
            usesMenuBarImage = false
        }
        item.button?.toolTip = appDisplayName

        let menu = NSMenu()
        let status = NSMenuItem(title: "Status: \(backend.state.label)", action: nil, keyEquivalent: "")
        status.isEnabled = false
        menu.addItem(status)
        let update = NSMenuItem(title: "Update Available", action: #selector(installUpdate), keyEquivalent: "")
        update.target = self
        update.isHidden = true
        menu.addItem(update)
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Open WakeWord Trainer", action: #selector(openTrainer), keyEquivalent: "o"))
        menu.addItem(NSMenuItem(title: "Open in Browser", action: #selector(openBrowser), keyEquivalent: "b"))
        menu.addItem(NSMenuItem.separator())
        let start = NSMenuItem(title: "Start", action: #selector(startTrainer), keyEquivalent: "s")
        let stop = NSMenuItem(title: "Stop", action: #selector(stopTrainer), keyEquivalent: "")
        menu.addItem(start)
        menu.addItem(NSMenuItem(title: "Restart", action: #selector(restartTrainer), keyEquivalent: "r"))
        menu.addItem(stop)
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Show Logs", action: #selector(showLogs), keyEquivalent: "l"))
        let checkUpdates = NSMenuItem(title: "Check for Updates...", action: #selector(checkForUpdates), keyEquivalent: "")
        checkUpdates.target = self
        menu.addItem(checkUpdates)
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit WakeWord Trainer", action: #selector(quit), keyEquivalent: "q"))

        item.menu = menu
        statusItem = item
        statusMenuItem = status
        startMenuItem = start
        stopMenuItem = stop
        updateMenuItem = update
        checkUpdatesMenuItem = checkUpdates
        refreshMenu(for: backend.state)
        refreshUpdateMenu(for: updater.state)
    }

    private func refreshMenu(for state: BackendState) {
        statusMenuItem?.title = "Status: \(state.label)"
        switch state {
        case .running, .starting, .bootstrapping:
            startMenuItem?.isEnabled = false
            stopMenuItem?.isEnabled = true
            if !usesMenuBarImage {
                statusItem?.button?.title = "W"
            }
        case .stopped, .failed:
            startMenuItem?.isEnabled = true
            stopMenuItem?.isEnabled = false
            if !usesMenuBarImage {
                statusItem?.button?.title = "W"
            }
        }
    }

    private func refreshUpdateMenu(for state: UpdateState) {
        updateMenuResetTimer?.invalidate()
        checkUpdatesMenuItem?.isEnabled = true
        checkUpdatesMenuItem?.title = "Check for Updates..."

        switch state {
        case .idle:
            hideUpdateItem()
        case .checking:
            hideUpdateItem()
            checkUpdatesMenuItem?.title = "Checking for Updates..."
            checkUpdatesMenuItem?.isEnabled = false
        case .current:
            hideUpdateItem()
            checkUpdatesMenuItem?.title = "WakeWord Trainer is Up to Date"
            resetCheckUpdateTitleSoon()
        case .available(let manifest):
            showOrangeUpdateItem("Update Available: \(manifest.version)", enabled: true)
        case .downloading(let manifest):
            showOrangeUpdateItem("Downloading WakeWord Trainer \(manifest.version)...", enabled: false)
            checkUpdatesMenuItem?.isEnabled = false
        case .installing(let manifest):
            showOrangeUpdateItem("Installing WakeWord Trainer \(manifest.version)...", enabled: false)
            checkUpdatesMenuItem?.isEnabled = false
        case .failed:
            hideUpdateItem()
            checkUpdatesMenuItem?.title = "Update Check Failed"
            resetCheckUpdateTitleSoon()
        }
    }

    private func showOrangeUpdateItem(_ title: String, enabled: Bool) {
        guard let updateMenuItem else { return }
        updateMenuItem.isHidden = false
        updateMenuItem.isEnabled = enabled
        updateMenuItem.attributedTitle = NSAttributedString(
            string: title,
            attributes: [
                .foregroundColor: NSColor.systemOrange,
                .font: NSFont.menuFont(ofSize: NSFont.systemFontSize)
            ]
        )
    }

    private func hideUpdateItem() {
        updateMenuItem?.isHidden = true
        updateMenuItem?.isEnabled = false
        updateMenuItem?.attributedTitle = nil
        updateMenuItem?.title = "Update Available"
    }

    private func resetCheckUpdateTitleSoon() {
        updateMenuResetTimer = Timer.scheduledTimer(withTimeInterval: 4.0, repeats: false) { [weak self] _ in
            self?.checkUpdatesMenuItem?.title = "Check for Updates..."
        }
    }

    private func resourceImage(named name: String, withExtension ext: String) -> NSImage? {
        guard let url = Bundle.main.url(forResource: name, withExtension: ext) else {
            return nil
        }
        return NSImage(contentsOf: url)
    }

    private func refreshWindow(for state: BackendState) {
        switch state {
        case .running:
            windowController?.setStatus("", webVisible: true)
            windowController?.load(url: backend.webURL)
        case .bootstrapping:
            windowController?.setStatus(
                "Setting up WakeWord Trainer",
                detail: "Preparing files and Python under \(backend.supportRoot.path)",
                webVisible: false
            )
        case .starting:
            windowController?.setStatus(
                "Starting WakeWord Trainer",
                detail: "Launching the local capture server on \(backend.webURL.absoluteString)",
                webVisible: false
            )
        case .stopped:
            windowController?.setStatus(
                "WakeWord Trainer is stopped",
                detail: "Use the menu bar item to start it again.",
                webVisible: false
            )
        case .failed(let message):
            windowController?.setStatus("WakeWord Trainer needs attention", detail: message, webVisible: false)
        }
    }

    @objc private func openTrainer() {
        showWindow()
    }

    @objc private func openBrowser() {
        NSWorkspace.shared.open(backend.webURL)
    }

    @objc private func startTrainer() {
        backend.start()
    }

    @objc private func stopTrainer() {
        backend.stop()
    }

    @objc private func restartTrainer() {
        backend.restart()
    }

    @objc private func showLogs() {
        backend.openLogsFolder()
    }

    @objc private func checkForUpdates() {
        updater.checkForUpdates(manual: true)
    }

    @objc private func installUpdate() {
        updater.installAvailableUpdate()
    }

    @objc private func quit() {
        NSApp.terminate(nil)
    }

    private func showWindow() {
        if windowController == nil {
            windowController = TrainerWindowController()
            windowController?.updateSetupProgress(from: backend.recentLogText())
        }
        windowController?.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
        refreshWindow(for: backend.state)
    }
}

private struct LauncherError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? {
        message
    }
}

private let app = NSApplication.shared
private let delegate = AppDelegate()
app.delegate = delegate
app.run()
