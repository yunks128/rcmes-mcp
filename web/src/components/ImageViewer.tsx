import { useEffect, useRef } from 'react';

interface ImageViewerProps {
  src: string | null;
  onClose: () => void;
}

export default function ImageViewer({ src, onClose }: ImageViewerProps) {
  const closeBtnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!src) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    closeBtnRef.current?.focus();
    return () => document.removeEventListener('keydown', onKey);
  }, [src, onClose]);

  if (!src) return null;

  return (
    <div
      className="image-viewer-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Image viewer"
    >
      <div className="image-viewer-content" onClick={e => e.stopPropagation()}>
        <button
          ref={closeBtnRef}
          className="image-viewer-close"
          onClick={onClose}
          aria-label="Close image viewer"
        >
          &times;
        </button>
        <img src={src} alt="Full size visualization" />
      </div>
    </div>
  );
}
