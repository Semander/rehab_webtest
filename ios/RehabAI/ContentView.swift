import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

struct ContentView: View {
    enum Source: String, CaseIterable, Identifiable {
        case live = "Live"
        case video = "Video"
        var id: String { rawValue }
    }

    @StateObject private var camera = CameraManager()
    @StateObject private var video = VideoPoseRunner()
    @State private var source: Source = .live
    @State private var pickerItem: PhotosPickerItem?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            background
            overlay
            VStack {
                picker
                if source == .video {
                    videoPicker
                }
                Spacer()
                if let message = activeError {
                    errorBanner(message)
                }
                anglePanel
            }
            .padding()
        }
        .onAppear { start(source) }
        .onDisappear {
            camera.stop()
            video.stop()
        }
        .onChange(of: source) { newValue in
            camera.stop()
            video.stop()
            start(newValue)
        }
        .onChange(of: pickerItem) { item in
            guard let item else { return }
            Task {
                if let movie = try? await item.loadTransferable(type: Movie.self) {
                    video.play(url: movie.url)
                }
            }
        }
    }


    @ViewBuilder private var background: some View {
        switch source {
        case .live:
            CameraPreviewView(session: camera.session)
                .ignoresSafeArea()
        case .video:
            if let image = video.displayImage {
                Image(decorative: image, scale: 1, orientation: .up)
                    .resizable()
                    .scaledToFit()
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "video.badge.plus")
                        .font(.system(size: 44))
                        .foregroundStyle(.secondary)
                    Text("Choose a video from your library")
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var overlay: some View {
        PoseOverlayView(pose: activePose,
                        imageSize: activeSource.imageSize,
                        contentMode: source == .live ? .fill : .fit)
            .ignoresSafeArea(edges: source == .live ? .all : [])
    }

    private var picker: some View {
        Picker("Source", selection: $source) {
            ForEach(Source.allCases) { Text($0.rawValue).tag($0) }
        }
        .pickerStyle(.segmented)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private var videoPicker: some View {
        PhotosPicker(selection: $pickerItem, matching: .videos) {
            Label("Choose Video", systemImage: "photo.on.rectangle.angled")
                .font(.subheadline.weight(.semibold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
        }
        .padding(.top, 4)
    }

    private struct AngleRow: Identifiable {
        let name: String
        let degrees: Double
        var id: String { name }
    }

    private var anglePanel: some View {
        let analyzer = activePose.map(PoseAnalyzer.init(pose:))
        let angles: [AngleRow] = [
            ("L elbow", analyzer?.leftElbow),
            ("R elbow", analyzer?.rightElbow),
            ("L shoulder", analyzer?.leftShoulder),
            ("R shoulder", analyzer?.rightShoulder),
        ].compactMap { name, value in value.map { AngleRow(name: name, degrees: $0) } }

        return VStack(alignment: .leading, spacing: 4) {
            Text(activePose?.isDetected == true ? "Joint angles" : "No pose detected")
                .font(.caption).foregroundStyle(.secondary)
            ForEach(angles) { angle in
                Text("\(angle.name): \(Int(angle.degrees.rounded()))°")
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.white)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.black.opacity(0.4), in: RoundedRectangle(cornerRadius: 12))
    }

    private func errorBanner(_ message: String) -> some View {
        Text(message)
            .font(.footnote)
            .foregroundStyle(.white)
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.red.opacity(0.8), in: RoundedRectangle(cornerRadius: 12))
    }

    private var activeSource: any PoseSource {
        source == .live ? camera : video
    }

    private var activePose: Pose? {
        source == .live ? camera.latestPose : video.latestPose
    }

    private var activeError: String? {
        source == .live ? camera.errorMessage : video.errorMessage
    }

    private func start(_ source: Source) {
        switch source {
        case .live: camera.start()
        case .video: video.start()
        }
    }
}
struct Movie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let copy = FileManager.default.temporaryDirectory
                .appendingPathComponent("picked_\(UUID().uuidString).mov")
            try? FileManager.default.removeItem(at: copy)
            try FileManager.default.copyItem(at: received.file, to: copy)
            return Movie(url: copy)
        }
    }
}
