// Automated Fingerprint Pattern Classification Frontend Application

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initDropzone();
  initModal();
  loadEvaluationReport();
});

// Tab Navigation
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabPanels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId)?.classList.add("active");
    });
  });
}

// Modal handling
function initModal() {
  const modal = document.getElementById("credits-modal");
  document.getElementById("btn-project-credits")?.addEventListener("click", () => {
    modal.classList.add("active");
  });
}

function closeModal() {
  document.getElementById("credits-modal")?.classList.remove("active");
}

// Dropzone & File Upload
// Dropzone & File Upload
function initDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const btnChange = document.getElementById("btn-change-file");

  dropzone.addEventListener("click", (e) => {
    if (e.target !== btnChange) {
      fileInput.click();
    }
  });

  btnChange?.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleLocalFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      handleLocalFile(fileInput.files[0]);
    }
  });
}

function handleLocalFile(file) {
  const isImage = file.type.startsWith("image/") || /\.(bmp|png|jpe?g|tif|tiff|webp)$/i.test(file.name);
  const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);

  const idleEl = document.getElementById("dropzone-idle-content");
  const loadedEl = document.getElementById("dropzone-loaded-content");
  const previewImg = document.getElementById("dropzone-preview-img");
  const filenameEl = document.getElementById("dropzone-filename");
  const metaEl = document.getElementById("dropzone-filemeta");

  if (idleEl) idleEl.style.display = "none";
  if (loadedEl) loadedEl.style.display = "block";
  if (filenameEl) filenameEl.textContent = file.name;

  const sizeKb = file.size < 1024 * 1024
    ? `${(file.size / 1024).toFixed(1)} KB`
    : `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  const ext = file.name.split('.').pop().toUpperCase();
  if (metaEl) metaEl.textContent = `${sizeKb} · ${ext} Biometric Scan`;

  if (isImage) {
    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImg) previewImg.src = e.target.result;
    };
    reader.readAsDataURL(file);
  } else if (isPdf) {
    if (previewImg) {
      previewImg.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 24 24' fill='none' stroke='%23ef4444' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpolyline points='14 2 14 8 20 8'/%3E%3Cline x1='16' y1='13' x2='8' y2='13'/%3E%3Cline x1='16' y1='17' x2='8' y2='17'/%3E%3Cpolyline points='10 9 9 9 8 9'/%3E%3C/svg%3E";
    }
  }

  uploadAndClassify(file);
}

async function uploadAndClassify(file) {
  setLoadingState(true, file.name);
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to process image.");
    }
    const data = await res.json();
    window._lastPredictionData = data;
    renderPredictionResult(data);
    showToast(
      "✅ Biometric Scan Analyzed",
      `${file.name} successfully classified as ${data.predicted_class_title} (${data.confidence_percentage})`
    );

    // Smooth scroll to results banner so user immediately sees results
    const banner = document.getElementById("upload-success-banner");
    if (banner) {
      banner.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  } catch (err) {
    showToast("❌ Analysis Error", err.message, true);
    alert("Inference Error: " + err.message);
  } finally {
    setLoadingState(false);
  }
}

async function runPreset(patternType) {
  setLoadingState(true, `${patternType.toUpperCase()} Preset`);
  try {
    const res = await fetch("/api/predict-preset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pattern_type: patternType }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to run preset inference.");
    }
    const data = await res.json();
    window._lastPredictionData = data;
    renderPredictionResult(data);
    showToast(
      "✅ Preset Loaded & Analyzed",
      `Sample print classified as ${data.predicted_class_title} (${data.confidence_percentage})`
    );
  } catch (err) {
    showToast("❌ Preset Error", err.message, true);
    alert("Preset Error: " + err.message);
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(isLoading, filename = "") {
  const title = document.getElementById("pattern-title");
  const desc = document.getElementById("pattern-desc");
  if (isLoading) {
    if (title) title.textContent = `Analyzing ${filename || 'Scan'}...`;
    if (desc) desc.textContent = "Executing 6-stage image preprocessing (Noise Filter, CLAHE, Otsu, Orientation Field) and CNN inference...";
  }
}

function showToast(title, desc, isError = false) {
  const toast = document.getElementById("toast-notification");
  const tTitle = document.getElementById("toast-title");
  const tDesc = document.getElementById("toast-desc");
  const tIcon = document.getElementById("toast-icon");
  if (!toast) return;

  if (tTitle) tTitle.textContent = title;
  if (tDesc) tDesc.textContent = desc;
  if (tIcon) tIcon.textContent = isError ? "⚠️" : "✅";

  if (isError) {
    toast.style.borderColor = "rgba(239, 68, 68, 0.5)";
    toast.style.boxShadow = "0 10px 30px rgba(0, 0, 0, 0.6), 0 0 20px rgba(239, 68, 68, 0.25)";
  } else {
    toast.style.borderColor = "rgba(16, 185, 129, 0.4)";
    toast.style.boxShadow = "0 10px 30px rgba(0, 0, 0, 0.6), 0 0 20px rgba(16, 185, 129, 0.25)";
  }

  toast.classList.add("show");
  clearTimeout(window._toastTimeout);
  window._toastTimeout = setTimeout(() => {
    toast.classList.remove("show");
  }, 4500);
}

function exportOfficialPdfReport() {
  window.print();
}

async function downloadBiometricPdfReport() {
  if (!window._lastPredictionData) {
    alert("Please upload or select a fingerprint scan first before downloading a report.");
    return;
  }

  showToast("⏳ Generating PDF", "Building official biometric examination report...");

  try {
    const res = await fetch("/api/export-pdf-report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(window._lastPredictionData),
    });

    if (!res.ok) {
      throw new Error("Failed to generate PDF report from server.");
    }

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    const filename = window._lastPredictionData.filename || "biometric_scan";
    const cleanName = filename.replace(/\.[^/.]+$/, "");
    a.href = url;
    a.download = `PU_Biometric_Report_${cleanName}.pdf`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();

    showToast("📥 Report Downloaded", `Saved PU_Biometric_Report_${cleanName}.pdf`);
  } catch (err) {
    showToast("❌ Download Failed", err.message, true);
    // Fallback to browser print if server PDF fails
    window.print();
  }
}

function renderPredictionResult(data) {
  // 1. Update Dropzone preview with the confirmed original image
  const idleEl = document.getElementById("dropzone-idle-content");
  const loadedEl = document.getElementById("dropzone-loaded-content");
  const dropzoneImg = document.getElementById("dropzone-preview-img");
  const filenameEl = document.getElementById("dropzone-filename");
  const metaEl = document.getElementById("dropzone-filemeta");

  if (idleEl) idleEl.style.display = "none";
  if (loadedEl) loadedEl.style.display = "block";
  if (data.preprocessing_stages && data.preprocessing_stages.original && dropzoneImg) {
    dropzoneImg.src = data.preprocessing_stages.original;
  }
  if (filenameEl) filenameEl.textContent = data.filename || "Uploaded Scan";
  if (metaEl) metaEl.textContent = `${data.filesize_formatted || '64.0 KB'} · ${data.file_type || 'BMP'} Biometric Scan`;

  // 2. Update Upload Verification Banner
  const banner = document.getElementById("upload-success-banner");
  const bannerIcon = document.getElementById("upload-banner-icon");
  const bannerStatus = document.getElementById("banner-file-status");
  const bannerPill = document.getElementById("banner-format-pill");
  const bannerMeta = document.getElementById("banner-meta-text");

  if (banner) {
    banner.style.display = "flex";
    if (bannerIcon) bannerIcon.textContent = data.pdf_info ? "📄" : "✅";
    if (bannerStatus) {
      bannerStatus.textContent = data.pdf_info
        ? "PDF Biometric Document Verified"
        : "Biometric Scan Verified & Ingested";
    }
    if (bannerPill) bannerPill.textContent = data.file_type || "SCAN";
    if (bannerMeta) {
      const extraInfo = data.pdf_info ? ` (${data.pdf_info})` : "";
      bannerMeta.textContent = `${data.filename || 'Scan'} · ${data.filesize_formatted || ''} · Latency: ${data.latency_ms || 32} ms${extraInfo}`;
    }
  }

  // 3. Update Hero section & File status indicator
  const badge = document.getElementById("pattern-badge");
  const title = document.getElementById("pattern-title");
  const desc = document.getElementById("pattern-desc");
  const fileIndicator = document.getElementById("hero-file-indicator");
  const filenameText = document.getElementById("hero-filename-text");
  const latencyBadge = document.getElementById("hero-latency-badge");
  const thumbBox = document.getElementById("hero-thumb-box");
  const heroThumb = document.getElementById("hero-scan-thumb");
  const pdfNote = document.getElementById("hero-pdf-note");
  const pdfText = document.getElementById("hero-pdf-text");

  if (fileIndicator) fileIndicator.style.display = "inline-flex";
  if (filenameText) filenameText.textContent = data.filename || "Analyzed Scan";
  if (latencyBadge) latencyBadge.textContent = `${data.latency_ms || 32} ms`;

  if (thumbBox && heroThumb && data.preprocessing_stages && data.preprocessing_stages.original) {
    thumbBox.style.display = "flex";
    heroThumb.src = data.preprocessing_stages.original;
  }

  if (data.pdf_info && pdfNote && pdfText) {
    pdfNote.style.display = "inline-flex";
    pdfText.textContent = data.pdf_info;
  } else if (pdfNote) {
    pdfNote.style.display = "none";
  }

  if (badge) {
    badge.textContent = `Predicted Pattern: ${data.predicted_class_title}`;
    badge.style.color = data.color_theme;
    badge.style.background = `${data.color_theme}22`;
    badge.style.border = `1px solid ${data.color_theme}44`;
  }

  if (title) title.textContent = data.predicted_class_title;
  if (desc) desc.textContent = `${data.anatomical_description} Key characteristics: ${data.key_features}`;

  document.getElementById("val-confidence").textContent = data.confidence_percentage;
  document.getElementById("val-runnerup").textContent = data.borderline_analysis?.runner_up_class?.toUpperCase() || "--";
  document.getElementById("val-prevalence").textContent = data.population_prevalence || "--";
  const valLatency = document.getElementById("val-latency");
  if (valLatency) valLatency.textContent = `${data.latency_ms || 32} ms`;

  // 4. Softmax probability bars
  const probs = data.probabilities;
  const pArch = (probs.arch || 0) * 100;
  const pLoop = (probs.loop || 0) * 100;
  const pWhorl = (probs.whorl || 0) * 100;

  document.getElementById("prob-val-arch").textContent = `${pArch.toFixed(1)}%`;
  document.getElementById("prob-val-loop").textContent = `${pLoop.toFixed(1)}%`;
  document.getElementById("prob-val-whorl").textContent = `${pWhorl.toFixed(1)}%`;

  document.getElementById("bar-arch").style.width = `${pArch}%`;
  document.getElementById("bar-loop").style.width = `${pLoop}%`;
  document.getElementById("bar-whorl").style.width = `${pWhorl}%`;

  // Borderline Near-Miss Box
  const alertContainer = document.getElementById("borderline-alert-container");
  const borderline = data.borderline_analysis;
  if (borderline && borderline.is_borderline) {
    alertContainer.innerHTML = `
      <div class="alert-box alert-warning">
        <span style="font-size:1.3rem;">⚠️</span>
        <div>
          <strong>Near-Miss / Borderline Pattern Detected (Slide 11 & 14)</strong>
          <p>${borderline.note}</p>
        </div>
      </div>
    `;
  } else if (borderline) {
    alertContainer.innerHTML = `
      <div class="alert-box alert-success">
        <span style="font-size:1.3rem;">✅</span>
        <div>
          <strong>High-Confidence Unambiguous Pattern</strong>
          <p>Sufficient decision boundary margin (${(borderline.margin * 100).toFixed(1)}%) separating ${data.predicted_class} from runner-up pattern.</p>
        </div>
      </div>
    `;
  }

  // Visual Preprocessing Stages
  if (data.preprocessing_stages) {
    const s = data.preprocessing_stages;
    if (s.original) document.getElementById("img-step-original").src = s.original;
    if (s.denoised) document.getElementById("img-step-denoised").src = s.denoised;
    if (s.enhanced) document.getElementById("img-step-enhanced").src = s.enhanced;
    if (s.mask) document.getElementById("img-step-mask").src = s.mask;
    if (s.segmented) document.getElementById("img-step-segmented").src = s.segmented;
    if (s.orientation) document.getElementById("img-step-orientation").src = s.orientation;
  }
}

// Evaluation Report Fetching
async function loadEvaluationReport() {
  const tableBody = document.getElementById("eval-table-body");
  const cmImg = document.getElementById("eval-cm-plot");
  try {
    const res = await fetch("/api/evaluation-report");
    if (!res.ok) throw new Error("Could not load report");
    const data = await res.json();

    document.getElementById("eval-accuracy").textContent = data.summary.accuracy_percentage;
    document.getElementById("eval-f1").textContent = `${(data.summary.f1_macro * 100).toFixed(1)}%`;

    if (data.confusion_matrix_plot) {
      cmImg.src = data.confusion_matrix_plot;
    }

    let rowsHtml = "";
    for (const [cname, metrics] of Object.entries(data.per_class)) {
      rowsHtml += `
        <tr>
          <td><strong style="text-transform:capitalize;">${cname}</strong></td>
          <td>${(metrics.precision * 100).toFixed(1)}%</td>
          <td>${(metrics.recall * 100).toFixed(1)}%</td>
          <td>${(metrics.f1_score * 100).toFixed(1)}%</td>
          <td>${metrics.support}</td>
        </tr>
      `;
    }
    tableBody.innerHTML = rowsHtml;
  } catch (err) {
    tableBody.innerHTML = `<tr><td colspan="5" style="color:#ef4444;">Evaluation report: ${err.message}</td></tr>`;
  }
}
