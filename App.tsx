import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  Upload, Download, Plus, Trash2, ZoomIn, ZoomOut,
  MousePointer, Square, Move, Brain, Settings, Layout,
  Image as ImageIcon, Check, Save, Package, CornerRightDown, Command, CircleHelp
} from 'lucide-react';
import JSZip from 'jszip';
import { Canvas } from './components/Canvas';
import { HelpModal } from './components/HelpModal';
import { DatasetImage, LabelClass, ToolMode, YOLOConfig, BBox } from './types';
import { DEFAULT_LABELS, DEFAULT_YOLO_CONFIG, COLORS } from './constants';
import { autoLabelImage } from './services/geminiService';
import { generateYoloAnnotation, generateDataYaml, generateTrainingScript } from './services/yoloService';

// --- ToolButton Component for UX ---
interface ToolButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  title: string;
  hotkey?: string;
  description: string;
  disabled?: boolean;
}

const ToolButton: React.FC<ToolButtonProps> = ({ active, onClick, icon, title, hotkey, description, disabled }) => {
  return (
    <div className="group relative flex justify-center">
      <button
        onClick={onClick}
        disabled={disabled}
        className={`p-2 rounded flex justify-center transition-all duration-200 ${active
          ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/50 ring-1 ring-blue-400/50'
          : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700 hover:text-neutral-200'
          } ${disabled ? 'opacity-30 cursor-not-allowed' : ''}`}
      >
        {icon}
      </button>

      {/* Tooltip */}
      <div className="absolute left-full top-0 ml-3 w-48 p-3 bg-neutral-900/95 backdrop-blur border border-neutral-700 rounded-lg shadow-2xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 pointer-events-none translate-x-[-10px] group-hover:translate-x-0">
        <div className="flex justify-between items-center mb-1">
          <span className="font-semibold text-sm text-white">{title}</span>
          {hotkey && <span className="text-[10px] bg-neutral-800 px-1.5 py-0.5 rounded border border-neutral-700 font-mono text-neutral-400">{hotkey}</span>}
        </div>
        <p className="text-xs text-neutral-400 leading-relaxed">{description}</p>
        {/* Arrow */}
        <div className="absolute top-3 -left-1 w-2 h-2 bg-neutral-900 border-l border-b border-neutral-700 transform rotate-45"></div>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  // State: Data
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null);
  const [labels, setLabels] = useState<LabelClass[]>(DEFAULT_LABELS);
  const [currentLabelId, setCurrentLabelId] = useState<string>(DEFAULT_LABELS[0].id);

  // State: UI
  const [toolMode, setToolMode] = useState<ToolMode>(ToolMode.SELECT);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [activeTab, setActiveTab] = useState<'editor' | 'train'>('editor');
  const [selectedAnnId, setSelectedAnnId] = useState<string | null>(null);
  const [isZipping, setIsZipping] = useState(false);
  const [showHelp, setShowHelp] = useState(false);

  // State: Config

  const [yoloConfig, setYoloConfig] = useState<YOLOConfig>(DEFAULT_YOLO_CONFIG);
  const [isProcessingAI, setIsProcessingAI] = useState(false);
  // Default to 2.5, but let user choose specifically if they want 1.5 or others
  const [aiModel, setAiModel] = useState<string>('gemini-2.5-flash');

  const editorRef = useRef<HTMLElement>(null);
  const currentImage = images.find(img => img.id === selectedImageId);

  // Effect: Auto-fit zoom for large images
  useEffect(() => {
    if (currentImage && editorRef.current && currentImage.width > 0 && currentImage.height > 0) {
      const { clientWidth, clientHeight } = editorRef.current;

      // If image is larger than the editor container, default to 50% zoom
      if (currentImage.width > clientWidth || currentImage.height > clientHeight) {
        setZoom(0.5);
      } else {
        setZoom(1);
      }
      // Reset pan
      setPan({ x: 0, y: 0 });
    }
  }, [currentImage?.id, currentImage?.width, currentImage?.height]);

  // Handlers
  const handleZipUpload = async (file: File) => {
    setIsZipping(true); // Use zipping loading state for import too
    try {
      const zip = new JSZip();
      const content = await zip.loadAsync(file);

      let newLabels: LabelClass[] = [];
      let activeLabels = labels;

      // 1. Look for data.yaml to restore classes
      const files = Object.values(content.files);
      const yamlFile = files.find(f => f.name.endsWith('data.yaml') && !f.dir && !f.name.includes('__MACOSX'));

      if (yamlFile) {
        const text = await yamlFile.async('string');
        const lines = text.split('\n');
        let inNames = false;
        const names: string[] = [];

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('names:')) {
            inNames = true;
            if (trimmed.includes('[')) {
              const content = trimmed.substring(trimmed.indexOf('[') + 1, trimmed.lastIndexOf(']'));
              const items = content.split(',').map(s => s.trim().replace(/['"]/g, ''));
              names.push(...items);
              inNames = false;
            }
            continue;
          }

          if (inNames) {
            if (trimmed.startsWith('-')) {
              names.push(trimmed.substring(1).trim());
            } else if (trimmed.includes(':') && !trimmed.startsWith('#') && !trimmed.endsWith(':')) {
              const parts = trimmed.split(':');
              if (parts.length >= 2) {
                names.push(parts[1].trim());
              } else {
                inNames = false;
              }
            } else if (trimmed === '') {
              continue;
            } else {
              inNames = false;
            }
          }
        }

        if (names.length > 0) {
          newLabels = names.map((name, idx) => ({
            id: idx.toString(),
            name,
            color: COLORS[idx % COLORS.length]
          }));
          setLabels(newLabels);
          activeLabels = newLabels;
          if (newLabels.length > 0) setCurrentLabelId(newLabels[0].id);
        }
      }

      // 2. Process Images and Annotations
      const newImages: DatasetImage[] = [];
      const labelMap = new Map<string, string>();

      // Pass 1: Index txt files
      for (const f of files) {
        if (!f.dir && f.name.endsWith('.txt') && !f.name.includes('__MACOSX') && !f.name.endsWith('classes.txt') && !f.name.endsWith('README.txt')) {
          const text = await f.async('string');
          const baseName = f.name.split('/').pop()?.replace(/\.txt$/, '');
          if (baseName) labelMap.set(baseName, text);
        }
      }

      // Pass 2: Process images
      for (const f of files) {
        if (!f.dir && f.name.match(/\.(jpg|jpeg|png|webp|bmp)$/i) && !f.name.includes('__MACOSX')) {
          const blob = await f.async('blob');
          const name = f.name.split('/').pop() || 'restored_image.jpg';
          const baseName = name.replace(/\.[^/.]+$/, "");

          const imgFile = new File([blob], name, { type: blob.type });
          const url = URL.createObjectURL(blob);

          const anns: BBox[] = [];
          const labelContent = labelMap.get(baseName);

          if (labelContent) {
            const lines = labelContent.split('\n');
            lines.forEach(line => {
              const parts = line.trim().split(/\s+/);
              if (parts.length >= 5) {
                const clsIdx = parseInt(parts[0]);
                const x = parseFloat(parts[1]);
                const y = parseFloat(parts[2]);
                const w = parseFloat(parts[3]);
                const h = parseFloat(parts[4]);

                const label = activeLabels[clsIdx];
                if (label) {
                  anns.push({
                    id: crypto.randomUUID(),
                    labelId: label.id,
                    x, y, w, h
                  });
                }
              }
            });
          }

          newImages.push({
            id: crypto.randomUUID(),
            file: imgFile,
            url,
            name,
            width: 0, // Will load async
            height: 0,
            annotations: anns,
            status: anns.length > 0 ? 'done' : 'unlabeled'
          });
        }
      }

      setImages(prev => [...prev, ...newImages]);

      newImages.forEach(img => {
        const i = new Image();
        i.src = img.url;
        i.onload = () => {
          setImages(prev => prev.map(p => p.id === img.id ? { ...p, width: i.naturalWidth, height: i.naturalHeight } : p));
        };
      });

      if (newImages.length > 0) {
        setSelectedImageId(newImages[0].id);
        alert(`Successfully imported ${newImages.length} images and dataset configuration.`);
      } else {
        alert("No valid images found in ZIP.");
      }

    } catch (e) {
      console.error(e);
      alert("Error processing ZIP file.");
    } finally {
      setIsZipping(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const filesValues = Array.from(e.target.files);
      const files: File[] = filesValues as File[];

      // Check if it's a zip file
      const zipFile = files.find(f => f.name.endsWith('.zip') || f.type.includes('zip') || f.type.includes('compressed'));
      if (zipFile) {
        handleZipUpload(zipFile);
        return;
      }

      // Standard image upload
      const newImages: DatasetImage[] = files.map(file => ({
        id: crypto.randomUUID(),
        file,
        url: URL.createObjectURL(file),
        name: file.name,
        width: 800,
        height: 600,
        annotations: [],
        status: 'unlabeled'
      }));

      newImages.forEach(img => {
        const i = new Image();
        i.src = img.url;
        i.onload = () => {
          setImages(prev => prev.map(p => p.id === img.id ? { ...p, width: i.naturalWidth, height: i.naturalHeight } : p));
        };
      });

      setImages(prev => [...prev, ...newImages]);
      if (!selectedImageId && newImages.length > 0) {
        setSelectedImageId(newImages[0].id);
      }
    }
  };

  const updateAnnotations = (newAnnotations: BBox[]) => {
    if (!selectedImageId) return;
    setImages(prev => prev.map(img => {
      if (img.id === selectedImageId) {
        const status = newAnnotations.length > 0 ? 'in-progress' : 'unlabeled';
        return { ...img, annotations: newAnnotations, status };
      }
      return img;
    }));
  };

  const handleDeleteAnnotation = () => {
    if (selectedAnnId && currentImage) {
      const newAnns = currentImage.annotations.filter(a => a.id !== selectedAnnId);
      updateAnnotations(newAnns);
      setSelectedAnnId(null);
    }
  };

  const handleAutoLabel = async () => {
    if (!currentImage) return;
    setIsProcessingAI(true);
    try {
      const newAnns = await autoLabelImage(currentImage, labels, { model: aiModel });
      updateAnnotations([...currentImage.annotations, ...newAnns]);
    } catch (err) {
      alert("AI Processing Failed. Check API Key or console.");
      console.error(err);
    } finally {
      setIsProcessingAI(false);
    }
  };

  const handleDownloadData = () => {
    // Simple download logic for single file
    if (activeTab === 'editor' && currentImage) {
      // Download single file
      const content = generateYoloAnnotation(currentImage.annotations, labels);
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = currentImage.name.replace(/\.[^/.]+$/, "") + ".txt";
      a.click();
    }
  };

  const handleDownloadFullDataset = async () => {
    if (images.length === 0) return;
    setIsZipping(true);

    try {
      const zip = new JSZip();
      const root = zip.folder("yolo_dataset");
      if (!root) throw new Error("Failed to create zip folder");

      // Structure:
      // /train/images
      // /train/labels
      // /val/images
      // /val/labels
      const trainImgs = root.folder("train")?.folder("images");
      const trainLbls = root.folder("train")?.folder("labels");
      const valImgs = root.folder("val")?.folder("images");
      const valLbls = root.folder("val")?.folder("labels");

      // We split 80/20 for train/val
      images.forEach((img, index) => {
        const isVal = index % 5 === 0; // Simple 20% validation split
        const targetImgFolder = isVal ? valImgs : trainImgs;
        const targetLblFolder = isVal ? valLbls : trainLbls;

        if (targetImgFolder && targetLblFolder) {
          // Add Image File
          targetImgFolder.file(img.name, img.file);

          // Add Annotation File
          // Even if empty, YOLO expects a file for negative samples, or we can skip.
          // Usually good to include empty txt for background images.
          const txtContent = generateYoloAnnotation(img.annotations, labels);
          const txtName = img.name.replace(/\.[^/.]+$/, "") + ".txt";
          targetLblFolder.file(txtName, txtContent);
        }
      });

      // Add Configuration Files
      root.file("data.yaml", generateDataYaml(labels));
      root.file("train.py", generateTrainingScript(yoloConfig));
      root.file("README.txt",
        "YOLOv11 Dataset Generated by YOLOv11 Studio\n" +
        "-------------------------------------------\n\n" +
        "Structure:\n" +
        "- train/images: Training images\n" +
        "- train/labels: Training labels\n" +
        "- val/images: Validation images\n" +
        "- val/labels: Validation labels\n" +
        "- data.yaml: YOLO configuration file\n" +
        "- train.py: Python training script\n\n" +
        "Usage:\n" +
        "1. Install ultralytics: pip install ultralytics\n" +
        "2. Run training: python train.py\n"
      );

      // Generate Blob
      const content = await zip.generateAsync({ type: "blob" });

      // Trigger Download
      const url = URL.createObjectURL(content);
      const a = document.createElement('a');
      a.href = url;
      a.download = `yolo11_dataset_${new Date().toISOString().slice(0, 10)}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

    } catch (error) {
      console.error("Export failed:", error);
      alert("Failed to create zip archive.");
    } finally {
      setIsZipping(false);
    }
  };

  // Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && activeTab === 'editor') handleDeleteAnnotation();
      if (e.key === 'v') setToolMode(ToolMode.SELECT);
      if (e.key === 'r') setToolMode(ToolMode.DRAW);
      if (e.key === 'h') setToolMode(ToolMode.PAN);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedAnnId, currentImage, activeTab]);

  return (
    <div className="flex flex-col h-screen w-full bg-neutral-900 text-white">
      {/* Help Modal */}
      <HelpModal isOpen={showHelp} onClose={() => setShowHelp(false)} />

      {/* Header */}
      <header className="h-14 border-b border-neutral-800 flex items-center justify-between px-4 bg-neutral-950 relative z-10">
        <div className="flex items-center gap-2">
          <Layout className="w-6 h-6 text-blue-500" />
          <h1 className="font-bold text-lg tracking-tight">YOLOv11 <span className="text-neutral-400 font-normal">Studio</span></h1>
        </div>
        <div className="flex bg-neutral-900 rounded-lg p-1 border border-neutral-800">
          <button
            onClick={() => setActiveTab('editor')}
            className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${activeTab === 'editor' ? 'bg-neutral-800 text-white' : 'text-neutral-400 hover:text-white'}`}
          >
            Editor
          </button>
          <button
            onClick={() => setActiveTab('train')}
            className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${activeTab === 'train' ? 'bg-neutral-800 text-white' : 'text-neutral-400 hover:text-white'}`}
          >
            Train / Export
          </button>
        </div>
        <div className="flex items-center gap-3">
          <a href="https://docs.ultralytics.com/" target="_blank" rel="noreferrer" className="text-xs text-neutral-500 hover:text-neutral-300 hidden md:block">
            Ultralytics Docs
          </a>
          <div className="w-px h-4 bg-neutral-800 hidden md:block"></div>
          <button
            onClick={() => setShowHelp(true)}
            className="p-2 text-neutral-400 hover:text-white hover:bg-neutral-800 rounded-full transition-colors"
            title="Help & Guide"
          >
            <CircleHelp size={20} />
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">

        {/* Left Sidebar (Tools & Labels) */}
        <aside className="w-64 border-r border-neutral-800 bg-neutral-900 flex flex-col z-10">

          {/* Tool Palette */}
          <div className="p-4 border-b border-neutral-800 grid grid-cols-4 gap-2">
            <ToolButton
              active={toolMode === ToolMode.SELECT}
              onClick={() => setToolMode(ToolMode.SELECT)}
              icon={<MousePointer size={18} />}
              title="Select Mode"
              hotkey="V"
              description="Select, move, and resize existing annotation boxes."
            />
            <ToolButton
              active={toolMode === ToolMode.DRAW}
              onClick={() => setToolMode(ToolMode.DRAW)}
              icon={<Square size={18} />}
              title="Draw Mode"
              hotkey="R"
              description="Click and drag on the canvas to create new bounding boxes."
            />
            <ToolButton
              active={toolMode === ToolMode.PAN}
              onClick={() => setToolMode(ToolMode.PAN)}
              icon={<Move size={18} />}
              title="Pan Tool"
              hotkey="H"
              description="Click and drag to move around the canvas while zoomed in."
            />
            <ToolButton
              active={false}
              onClick={handleDeleteAnnotation}
              disabled={!selectedAnnId}
              icon={<Trash2 size={18} />}
              title="Delete"
              hotkey="Del"
              description="Remove the currently selected annotation."
            />
          </div>

          {/* Classes List */}
          <div className="flex-1 overflow-y-auto p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-neutral-400 uppercase tracking-wider">Classes</h3>
              <button className="p-1 hover:bg-neutral-800 rounded transition-colors" onClick={() => {
                const newId = labels.length.toString();
                setLabels([...labels, { id: newId, name: 'new_class', color: COLORS[labels.length % COLORS.length] }]);
              }}>
                <Plus size={16} />
              </button>
            </div>
            <div className="space-y-2">
              {labels.map(label => {
                // Calculate annotation count for this label across all images
                const count = images.reduce((acc, img) =>
                  acc + img.annotations.filter(a => a.labelId === label.id).length, 0
                );

                return (
                  <div
                    key={label.id}
                    onClick={() => setCurrentLabelId(label.id)}
                    className={`flex items-center p-2 rounded-md cursor-pointer border transition-all ${currentLabelId === label.id
                      ? 'bg-neutral-800 border-neutral-700 shadow-sm'
                      : 'border-transparent hover:bg-neutral-800/50'
                      }`}
                  >
                    <div className="w-3 h-3 rounded-full mr-3 shadow-sm" style={{ backgroundColor: label.color }}></div>
                    <input
                      className="bg-transparent text-sm w-full focus:outline-none text-neutral-200 placeholder-neutral-600"
                      value={label.name}
                      onChange={(e) => {
                        const val = e.target.value;
                        setLabels(l => l.map(x => x.id === label.id ? { ...x, name: val } : x));
                      }}
                    />
                    <div className="flex items-center gap-2 min-w-fit">
                      <span className="text-[10px] text-neutral-600 font-mono">#{label.id}</span>
                      <span
                        className="text-[10px] bg-neutral-950 text-neutral-400 px-1.5 py-0.5 rounded-full border border-neutral-800 min-w-[1.5rem] text-center"
                        title={`${count} annotations`}
                      >
                        {count}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* AI Assistant */}
          <div className="p-4 border-t border-neutral-800">
            <button
              disabled={!currentImage || isProcessingAI}
              onClick={handleAutoLabel}
              className="w-full py-3 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg flex items-center justify-center gap-2 font-medium shadow-lg shadow-indigo-900/20 hover:shadow-indigo-900/40 hover:scale-[1.02] active:scale-[0.98] transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 mb-3"
            >
              {isProcessingAI ? (
                <span className="animate-pulse">Thinking...</span>
              ) : (
                <>
                  <Brain size={18} />
                  <span>AI Auto-Label</span>
                </>
              )}
            </button>

            {/* Model Selector */}
            <div className="flex items-center justify-between text-xs px-1">
              <span className="text-neutral-500">Model:</span>
              <select
                value={aiModel}
                onChange={(e) => setAiModel(e.target.value)}
                className="bg-transparent text-neutral-400 border-b border-neutral-700 pb-0.5 focus:outline-none focus:border-blue-500 focus:text-neutral-300 transition-colors text-right max-w-[120px]"
              >
                <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                <option value="gemini-2.0-flash">Gemini 2.0 Flash</option>
                <option value="gemini-1.5-flash">Gemini 1.5 Flash</option>
                <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
              </select>
            </div>
          </div>
        </aside>

        {/* Center Workspace */}
        <main className="flex-1 bg-neutral-950 relative flex flex-col" ref={editorRef}>
          {activeTab === 'editor' ? (
            <>
              {/* Toolbar Overlay */}
              <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-neutral-800/90 backdrop-blur border border-neutral-700 p-1.5 rounded-full flex gap-2 z-20 shadow-xl">
                <button className="p-2 hover:bg-white/10 rounded-full" onClick={() => setZoom(z => Math.max(0.1, z - 0.1))}><ZoomOut size={16} /></button>
                <span className="px-2 flex items-center text-xs font-mono min-w-[3rem] justify-center">{Math.round(zoom * 100)}%</span>
                <button className="p-2 hover:bg-white/10 rounded-full" onClick={() => setZoom(z => z + 0.1)}><ZoomIn size={16} /></button>
                <div className="w-px bg-neutral-600 mx-1 h-6 my-auto"></div>
                <button className="p-2 hover:bg-white/10 rounded-full" onClick={() => setPan({ x: 0, y: 0 })} title="Reset View"><Layout size={16} /></button>
              </div>

              {/* Canvas */}
              {currentImage ? (
                <Canvas
                  image={currentImage}
                  currentLabelId={currentLabelId}
                  labels={labels}
                  mode={toolMode}
                  zoom={zoom}
                  pan={pan}
                  onUpdateAnnotations={updateAnnotations}
                  onPanChange={(x, y) => setPan({ x, y })}
                  onSelectAnnotation={setSelectedAnnId}
                  selectedAnnotationId={selectedAnnId}
                />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center text-neutral-500">
                  <ImageIcon size={48} className="mb-4 opacity-20" />
                  <p>No image selected</p>
                  <p className="text-sm opacity-50">Upload images to begin</p>
                </div>
              )}
            </>
          ) : (
            // Training / Export Tab
            <div className="p-8 max-w-4xl mx-auto w-full overflow-y-auto">
              <h2 className="text-2xl font-bold mb-6">Export & Train Configuration</h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-6">
                  <div className="bg-neutral-900 p-6 rounded-xl border border-neutral-800">
                    <h3 className="font-semibold mb-4 flex items-center gap-2"><Settings size={18} /> YOLOv11 Hyperparameters</h3>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-xs text-neutral-400 mb-1">Epochs</label>
                        <input type="number" className="w-full bg-neutral-950 border border-neutral-800 rounded p-2 focus:border-blue-500 outline-none transition-colors" value={yoloConfig.epochs} onChange={e => setYoloConfig({ ...yoloConfig, epochs: +e.target.value })} />
                      </div>
                      <div>
                        <label className="block text-xs text-neutral-400 mb-1">Image Size (px)</label>
                        <input type="number" className="w-full bg-neutral-950 border border-neutral-800 rounded p-2 focus:border-blue-500 outline-none transition-colors" value={yoloConfig.imgSize} onChange={e => setYoloConfig({ ...yoloConfig, imgSize: +e.target.value })} />
                      </div>
                      <div>
                        <label className="block text-xs text-neutral-400 mb-1">Batch Size</label>
                        <input type="number" className="w-full bg-neutral-950 border border-neutral-800 rounded p-2 focus:border-blue-500 outline-none transition-colors" value={yoloConfig.batchSize} onChange={e => setYoloConfig({ ...yoloConfig, batchSize: +e.target.value })} />
                      </div>
                    </div>
                  </div>

                  <div className="bg-neutral-900 p-6 rounded-xl border border-neutral-800">
                    <h3 className="font-semibold mb-4 flex items-center gap-2"><Download size={18} /> Dataset Export</h3>
                    <p className="text-sm text-neutral-400 mb-4">Export your dataset. The ZIP includes images (split 80/20 for train/val), labels, data.yaml, and training scripts.</p>
                    <div className="flex gap-3 flex-col">
                      <button
                        onClick={handleDownloadData}
                        disabled={!currentImage}
                        className="bg-neutral-800 hover:bg-neutral-700 px-4 py-3 rounded text-sm font-medium disabled:opacity-50 flex items-center justify-between transition-colors"
                      >
                        <span>Download Current Image Labels (.txt)</span>
                        <Download size={16} />
                      </button>

                      <button
                        onClick={handleDownloadFullDataset}
                        disabled={images.length === 0 || isZipping}
                        className="bg-blue-600 hover:bg-blue-500 px-4 py-3 rounded text-sm font-medium disabled:opacity-50 flex items-center justify-between shadow-lg shadow-blue-900/20 transition-all"
                      >
                        <div className="flex flex-col items-start">
                          <span>Download Full Dataset (.zip)</span>
                          <span className="text-[10px] opacity-70 font-normal">Includes images, labels, yaml & script</span>
                        </div>
                        {isZipping ? <span className="animate-spin">⏳</span> : <Package size={18} />}
                      </button>
                    </div>
                  </div>
                </div>

                <div className="space-y-6">
                  <div className="bg-neutral-900 p-6 rounded-xl border border-neutral-800">
                    <h3 className="font-semibold mb-4">Generated data.yaml</h3>
                    <pre className="bg-black/50 p-4 rounded text-xs text-green-400 font-mono overflow-x-auto border border-neutral-800">
                      {generateDataYaml(labels)}
                    </pre>
                  </div>

                  <div className="bg-neutral-900 p-6 rounded-xl border border-neutral-800">
                    <h3 className="font-semibold mb-4">Python Training Script</h3>
                    <pre className="bg-black/50 p-4 rounded text-xs text-blue-400 font-mono overflow-x-auto border border-neutral-800">
                      {generateTrainingScript(yoloConfig)}
                    </pre>
                    <button
                      className="mt-4 w-full py-2 border border-neutral-700 rounded hover:bg-neutral-800 text-xs uppercase tracking-wider transition-colors"
                      onClick={() => navigator.clipboard.writeText(generateTrainingScript(yoloConfig))}
                    >
                      Copy to Clipboard
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>

        {/* Right Sidebar (Images) */}
        <aside className="w-64 border-l border-neutral-800 bg-neutral-900 flex flex-col z-10">
          <div className="p-4 border-b border-neutral-800">
            <h3 className="font-semibold text-sm mb-3 text-neutral-300">Dataset Images</h3>
            <label className="flex items-center justify-center w-full p-2 bg-neutral-800 hover:bg-neutral-700 rounded border border-dashed border-neutral-600 cursor-pointer transition-all hover:border-neutral-500 group">
              <Upload size={16} className="mr-2 text-neutral-400 group-hover:text-white transition-colors" />
              <span className="text-xs font-medium text-neutral-300 group-hover:text-white transition-colors">Upload Images / ZIP</span>
              <input type="file" multiple accept="image/*,.zip,application/zip,application/x-zip-compressed" className="hidden" onChange={handleFileUpload} />
            </label>
          </div>

          <div className="flex-1 overflow-y-auto">
            {images.length === 0 ? (
              <div className="p-4 text-center text-xs text-neutral-600 mt-10">
                No images uploaded yet.
              </div>
            ) : (
              <div className="divide-y divide-neutral-800">
                {images.map((img, idx) => (
                  <div
                    key={img.id}
                    onClick={() => {
                      setSelectedImageId(img.id);
                    }}
                    className={`p-3 flex items-start gap-3 cursor-pointer transition-colors ${selectedImageId === img.id ? 'bg-blue-900/20 border-l-2 border-blue-500' : 'hover:bg-neutral-800 border-l-2 border-transparent'}`}
                  >
                    <div className="w-12 h-12 bg-neutral-800 rounded overflow-hidden shrink-0 relative">
                      <img src={img.url} className="w-full h-full object-cover opacity-80" alt="thumb" />
                    </div>
                    <div className="overflow-hidden">
                      <div className="text-xs font-medium truncate text-neutral-300 mb-1" title={img.name}>{img.name}</div>
                      <div className="flex items-center gap-2">
                        <span className={`text-[10px] px-1.5 py-0.5 rounded uppercase tracking-wider ${img.status === 'done' ? 'bg-green-900 text-green-400' :
                          img.status === 'in-progress' ? 'bg-yellow-900 text-yellow-400' : 'bg-neutral-800 text-neutral-500'
                          }`}>
                          {img.status === 'in-progress' ? 'Labeling' : img.status}
                        </span>
                        <span className="text-[10px] text-neutral-500">{img.annotations.length} box</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Stats Footer */}
          <div className="p-3 border-t border-neutral-800 bg-neutral-950 text-xs text-neutral-500 flex justify-between">
            <span>{images.length} Images</span>
            <span>{images.reduce((acc, cur) => acc + cur.annotations.length, 0)} Annotations</span>
          </div>
        </aside>
      </div>
    </div>
  );
};

export default App;