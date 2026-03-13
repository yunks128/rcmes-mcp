interface ImageViewerProps {
  src: string | null;
  onClose: () => void;
}

export default function ImageViewer({ src, onClose }: ImageViewerProps) {
  if (!src) return null;

  return (
    <div className="image-viewer-overlay" onClick={onClose}>
      <div className="image-viewer-content" onClick={e => e.stopPropagation()}>
        <button className="image-viewer-close" onClick={onClose}>
          &times;
        </button>
        <img src={src} alt="Full size visualization" />
      </div>
    </div>
  );
}
