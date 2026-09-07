import { useEffect, useRef, useState } from "react";
import { EXPORT_FORMATS, type ExportFormat } from "../export";

/**
 * An "Export" button that opens a dropdown of formats (JSON, CSV, PDF).
 *
 * Closes when a format is chosen, when focus leaves via Escape, or on a click
 * outside -- the behaviour a dropdown is expected to have.
 */
export function ExportMenu({
  onExport,
  disabled,
}: {
  onExport: (format: ExportFormat) => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    const onPointerDown = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };

    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div className="export" ref={root}>
      <button
        type="button"
        className="btn btn--ghost"
        onClick={() => setOpen((was) => !was)}
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        Export ▾
      </button>

      {open && (
        <ul className="export__menu" role="menu">
          {EXPORT_FORMATS.map((format) => (
            <li key={format} role="none">
              <button
                type="button"
                role="menuitem"
                className="export__item"
                onClick={() => {
                  onExport(format);
                  setOpen(false);
                }}
              >
                {format.toUpperCase()}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
