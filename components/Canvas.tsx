import React, { useRef, useState, MouseEvent } from 'react';
import { BBox, DatasetImage, LabelClass, ToolMode } from '../types';
import { yoloToSvg, svgToYolo } from '../services/yoloService';

interface CanvasProps {
  image: DatasetImage;
  currentLabelId: string;
  labels: LabelClass[];
  mode: ToolMode;
  zoom: number;
  pan: { x: number; y: number };
  onUpdateAnnotations: (annotations: BBox[]) => void;
  onPanChange: (x: number, y: number) => void;
  onSelectAnnotation: (id: string | null) => void;
  selectedAnnotationId: string | null;
}

type ResizeHandleType = 'tl' | 'tr' | 'bl' | 'br';

export const Canvas: React.FC<CanvasProps> = ({
  image,
  currentLabelId,
  labels,
  mode,
  zoom,
  pan,
  onUpdateAnnotations,
  onPanChange,
  onSelectAnnotation,
  selectedAnnotationId
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [activeCreation, setActiveCreation] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [draggingAnnotationId, setDraggingAnnotationId] = useState<string | null>(null);
  const [resizeHandle, setResizeHandle] = useState<ResizeHandleType | null>(null);

  // Mouse events for the drawing area
  const getMousePos = (e: MouseEvent) => {
    if (!containerRef.current) return { x: 0, y: 0 };
    const rect = containerRef.current.getBoundingClientRect();
    // Adjust for zoom and pan
    const x = (e.clientX - rect.left - pan.x) / zoom;
    const y = (e.clientY - rect.top - pan.y) / zoom;
    return { x, y };
  };

  const handleMouseDown = (e: MouseEvent) => {
    const { x, y } = getMousePos(e);

    if (mode === ToolMode.PAN) {
      setIsDragging(true);
      setDragStart({ x: e.clientX, y: e.clientY });
      return;
    }

    // If clicking on empty space
    if (mode === ToolMode.DRAW) {
      setIsDragging(true);
      setActiveCreation({ x, y, w: 0, h: 0 });
      setDragStart({ x, y });
      onSelectAnnotation(null);
    } else if (mode === ToolMode.SELECT) {
      // Logic handled in SVG elements onMouseDown, but if we click empty space:
      onSelectAnnotation(null);
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    // 1. Handle Panning
    if (mode === ToolMode.PAN && isDragging && dragStart) {
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      onPanChange(pan.x + dx, pan.y + dy);
      setDragStart({ x: e.clientX, y: e.clientY });
      return;
    }

    const { x, y } = getMousePos(e);

    // 2. Handle Creating New Box
    if (mode === ToolMode.DRAW && isDragging && dragStart) {
      const w = x - dragStart.x;
      const h = y - dragStart.y;
      setActiveCreation({
        x: w > 0 ? dragStart.x : x,
        y: h > 0 ? dragStart.y : y,
        w: Math.abs(w),
        h: Math.abs(h)
      });
      return;
    }

    // 3. Handle Resizing Existing Box
    if (mode === ToolMode.SELECT && resizeHandle && selectedAnnotationId) {
      const ann = image.annotations.find(a => a.id === selectedAnnotationId);
      if (ann) {
        const rect = yoloToSvg(ann, image.width, image.height);
        let newX = rect.x;
        let newY = rect.y;
        let newW = rect.w;
        let newH = rect.h;

        // Calculate bounds based on handle type
        // We use the *opposite* corner as the anchor point
        if (resizeHandle === 'tl') {
          const brX = rect.x + rect.w;
          const brY = rect.y + rect.h;
          newX = Math.min(x, brX - 5); // Enforce min width
          newY = Math.min(y, brY - 5);
          newW = brX - newX;
          newH = brY - newY;
        } else if (resizeHandle === 'tr') {
          const blX = rect.x;
          const blY = rect.y + rect.h;
          newY = Math.min(y, blY - 5);
          newW = Math.max(x - blX, 5);
          newH = blY - newY;
        } else if (resizeHandle === 'bl') {
          const trX = rect.x + rect.w;
          const trY = rect.y;
          newX = Math.min(x, trX - 5);
          newW = trX - newX;
          newH = Math.max(y - trY, 5);
        } else if (resizeHandle === 'br') {
          const tlX = rect.x;
          const tlY = rect.y;
          newW = Math.max(x - tlX, 5);
          newH = Math.max(y - tlY, 5);
        }

        // Update annotation
        const newYolo = svgToYolo(newX, newY, newW, newH, image.width, image.height);
        const updatedAnns = image.annotations.map(a => 
          a.id === ann.id ? { ...a, ...newYolo } : a
        );
        onUpdateAnnotations(updatedAnns);
      }
      return;
    }

    // 4. Handle Moving Existing Box
    if (mode === ToolMode.SELECT && isDragging && draggingAnnotationId) {
      const ann = image.annotations.find(a => a.id === draggingAnnotationId);
      if (ann && dragStart) {
        // Calculate delta in normalized coords
        const dx = (x - dragStart.x) / image.width;
        const dy = (y - dragStart.y) / image.height;
        
        const newAnns = image.annotations.map(a => {
          if (a.id === draggingAnnotationId) {
            // Simple clamp to keep inside image (optional)
            return { ...a, x: a.x + dx, y: a.y + dy };
          }
          return a;
        });
        
        onUpdateAnnotations(newAnns);
        setDragStart({ x, y }); // Reset start to current to avoid accumulation errors
      }
    }
  };

  const handleMouseUp = () => {
    // Finalize Creation
    if (mode === ToolMode.DRAW && activeCreation) {
      if (activeCreation.w > 5 && activeCreation.h > 5) {
        const yoloBox = svgToYolo(
          activeCreation.x, activeCreation.y, 
          activeCreation.w, activeCreation.h, 
          image.width, image.height
        );
        
        const newAnn: BBox = {
          ...yoloBox,
          id: crypto.randomUUID(),
          labelId: currentLabelId
        };
        onUpdateAnnotations([...image.annotations, newAnn]);
        onSelectAnnotation(newAnn.id);
      }
      setActiveCreation(null);
    }

    setIsDragging(false);
    setDragStart(null);
    setDraggingAnnotationId(null);
    setResizeHandle(null);
  };

  // Annotation Interactions
  const handleAnnMouseDown = (e: MouseEvent, id: string) => {
    e.stopPropagation();
    if (mode === ToolMode.SELECT) {
      onSelectAnnotation(id);
      setIsDragging(true);
      const { x, y } = getMousePos(e);
      setDragStart({ x, y });
      setDraggingAnnotationId(id);
    }
  };

  const handleResizeStart = (e: MouseEvent, handle: ResizeHandleType) => {
    e.stopPropagation(); // Don't trigger box move or canvas click
    setResizeHandle(handle);
    setIsDragging(true);
  };

  return (
    <div 
      className="relative w-full h-full bg-neutral-900 overflow-hidden cursor-crosshair"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <div
        ref={containerRef}
        className="absolute origin-top-left"
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          width: image.width,
          height: image.height
        }}
        onMouseDown={handleMouseDown}
      >
        <img 
          src={image.url} 
          alt="workspace" 
          className="absolute top-0 left-0 select-none pointer-events-none"
          style={{ width: '100%', height: '100%' }}
          draggable={false}
        />

        <svg 
          width={image.width} 
          height={image.height} 
          className="absolute top-0 left-0"
        >
          {image.annotations.map(ann => {
            const { x, y, w, h } = yoloToSvg(ann, image.width, image.height);
            const label = labels.find(l => l.id === ann.labelId);
            const isSelected = selectedAnnotationId === ann.id;
            const color = label?.color || '#1d4ed8'; // Deep blue default

            return (
              <g key={ann.id}>
                <rect
                  x={x} y={y} width={w} height={h}
                  fill={color}
                  fillOpacity={isSelected ? 0.2 : 0.05}
                  stroke={color}
                  strokeWidth={isSelected ? 2 / zoom : 1.5 / zoom}
                  onMouseDown={(e) => handleAnnMouseDown(e, ann.id)}
                  className={`transition-all ${mode === ToolMode.SELECT ? 'cursor-move' : ''}`}
                />
                 {/* Label Text */}
                 <text
                    x={x}
                    y={y - 5 / zoom}
                    fill={color}
                    fontSize={14 / zoom}
                    fontWeight="bold"
                    className="select-none pointer-events-none"
                    style={{ textShadow: '0px 1px 2px black' }}
                  >
                    {label?.name}
                  </text>
                
                {isSelected && mode === ToolMode.SELECT && (
                  <>
                    {/* Resize Handles */}
                    {/* Top Left */}
                    <circle 
                      cx={x} cy={y} r={5/zoom} 
                      fill="white" stroke={color} strokeWidth={1/zoom} 
                      className="cursor-nwse-resize"
                      onMouseDown={(e) => handleResizeStart(e, 'tl')}
                    />
                    {/* Top Right */}
                    <circle 
                      cx={x+w} cy={y} r={5/zoom} 
                      fill="white" stroke={color} strokeWidth={1/zoom} 
                      className="cursor-nesw-resize"
                      onMouseDown={(e) => handleResizeStart(e, 'tr')}
                    />
                    {/* Bottom Left */}
                    <circle 
                      cx={x} cy={y+h} r={5/zoom} 
                      fill="white" stroke={color} strokeWidth={1/zoom} 
                      className="cursor-nesw-resize"
                      onMouseDown={(e) => handleResizeStart(e, 'bl')}
                    />
                    {/* Bottom Right */}
                    <circle 
                      cx={x+w} cy={y+h} r={5/zoom} 
                      fill="white" stroke={color} strokeWidth={1/zoom} 
                      className="cursor-nwse-resize"
                      onMouseDown={(e) => handleResizeStart(e, 'br')}
                    />
                  </>
                )}
              </g>
            );
          })}

          {activeCreation && mode === ToolMode.DRAW && (
             <rect
             x={activeCreation.x} y={activeCreation.y} 
             width={activeCreation.w} height={activeCreation.h}
             fill="rgba(37, 99, 235, 0.3)"
             stroke="#1d4ed8"
             strokeWidth={2 / zoom}
             strokeDasharray="4 2"
           />
          )}
        </svg>
      </div>
    </div>
  );
};
