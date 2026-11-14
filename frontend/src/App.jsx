
import { useState } from "react";
import {
  Upload,
  Image as ImageIcon,
  ShieldCheck,
  AlertTriangle,
  Loader2,
  X,
  ScanSearch,
} from "lucide-react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [model, setModel] = useState("smaug");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    if (!["image/jpeg", "image/png"].includes(selectedFile.type)) {
      setError("Please upload a JPG or PNG image.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Image size must be less than 10 MB.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };

  const handleDrop = (event) => {
    event.preventDefault();
    handleFile(event.dataTransfer.files[0]);
  };

  const clearImage = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
  };

  const analyzeImage = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `${API_URL}/predict?model_name=${model}`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Image analysis failed."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to the detection server."
      );
    } finally {
      setLoading(false);
    }
  };

  const confidence = result
    ? Number(result.confidence) * 100
    : 0;

  const isFake = result?.prediction === "FAKE";

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <ShieldCheck size={25} />
          </div>
          <div>
            <h2>ImageGuard</h2>
            <span>AI Image Detection</span>
          </div>
        </div>

        <div className="status">
          <span className="status-dot" />
          Detection platform
        </div>
      </header>

      <main className="main-container">
        <section className="hero">
          <div className="eyebrow">
            <ScanSearch size={15} />
            IMAGE AUTHENTICITY ANALYSIS
          </div>

          <h1>
            Is this image <span>real or AI?</span>
          </h1>

          <p>
            Upload an image and analyze it using deep learning
            models to identify potential AI-generated content.
          </p>
        </section>

        <section className="workspace">
          <div className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <h3>Upload image</h3>
                <p>Select an image for analysis</p>
              </div>
              <ImageIcon size={22} />
            </div>

            {!preview ? (
              <label
                className="dropzone"
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  hidden
                  onChange={(e) =>
                    handleFile(e.target.files[0])
                  }
                />

                <div className="upload-icon">
                  <Upload size={28} />
                </div>

                <h3>Drop your image here</h3>
                <p>or click to browse files</p>
                <span>JPG or PNG · Maximum 10 MB</span>
              </label>
            ) : (
              <div className="preview-container">
                <img
                  src={preview}
                  alt="Uploaded image preview"
                  className="preview-image"
                />

                <button
                  className="remove-button"
                  onClick={clearImage}
                  title="Remove image"
                >
                  <X size={18} />
                </button>

                <div className="file-details">
                  <ImageIcon size={18} />
                  <div>
                    <strong>{file.name}</strong>
                    <span>
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div className="model-selection">
              <label htmlFor="model">Select detection model</label>

              <select
                id="model"
                value={model}
                onChange={(e) => {
                  setModel(e.target.value);
                  setResult(null);
                }}
              >
                <option value="smaug">
                  Smaug – High-resolution detector
                </option>
                <option value="efficientnet">
                  EfficientNet-B0 – Original model
                </option>
              </select>

              <button
                className="analyze-button"
                onClick={analyzeImage}
                disabled={!file || loading}
              >
                {loading ? (
                  <>
                    <Loader2 className="spin" size={19} />
                    Analyzing image...
                  </>
                ) : (
                  <>
                    <ScanSearch size={19} />
                    Analyze image
                  </>
                )}
              </button>
            </div>

            {error && (
              <div className="error-message">
                <AlertTriangle size={18} />
                {error}
              </div>
            )}
          </div>

          <div className="panel result-panel">
            <div className="panel-heading">
              <div>
                <h3>Detection results</h3>
                <p>Analysis output and confidence scores</p>
              </div>
              <ShieldCheck size={22} />
            </div>

            {!result ? (
              <div className="empty-result">
                <div className="empty-icon">
                  <ScanSearch size={34} />
                </div>
                <h3>Waiting for analysis</h3>
                <p>
                  Upload an image and click Analyze image
                  to see the detection results here.
                </p>
              </div>
            ) : (
              <div className="result-content">
                <div
                  className={`prediction-card ${
                    isFake ? "fake" : "real"
                  }`}
                >
                  {isFake ? (
                    <AlertTriangle size={30} />
                  ) : (
                    <ShieldCheck size={30} />
                  )}

                  <div>
                    <span>DETECTION RESULT</span>
                    <h2>{result.prediction}</h2>
                    <p>
                      {isFake
                        ? "The detector identifies this image as AI-generated."
                        : "The detector classifies this image as real."}
                    </p>
                  </div>
                </div>

                <div className="confidence-section">
                  <div className="confidence-heading">
                    <span>Detector score</span>
                    <strong>{confidence.toFixed(2)}%</strong>
                  </div>

                  <div className="progress-track">
                    <div
                      className={`progress-fill ${
                        isFake ? "fake-fill" : "real-fill"
                      }`}
                      style={{ width: `${confidence}%` }}
                    />
                  </div>
                  <p>
                    Confidence reported by the selected detector;
                    not a guarantee of authenticity.
                  </p>
                </div>

                <div className="score-grid">
                  <div className="score-card">
                    <span>AI-generated score</span>
                    <strong>
                      {(
                        Number(result.class_scores?.FAKE ?? 0) * 100
                      ).toFixed(2)}%
                    </strong>
                  </div>

                  <div className="score-card">
                    <span>Real score</span>
                    <strong>
                      {(
                        Number(result.class_scores?.REAL ?? 0) * 100
                      ).toFixed(2)}%
                    </strong>
                  </div>
                </div>

                <div className="metadata">
                  <div>
                    <span>Model used</span>
                    <strong>{result.model_used}</strong>
                  </div>
                  <div>
                    <span>File name</span>
                    <strong>{result.filename}</strong>
                  </div>
                </div>

                <div className="disclaimer">
                  <AlertTriangle size={17} />
                  This is an automated prediction, not proof
                  that an image is authentic or manipulated.
                </div>
              </div>
            )}
          </div>
        </section>

        <footer>
          ImageGuard · Deepfake Image Detection System
        </footer>
      </main>
    </div>
  );
}

export default App;
