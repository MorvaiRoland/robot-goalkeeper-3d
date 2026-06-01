import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                             QLabel, QLineEdit, QPushButton, QGroupBox, QComboBox,
                             QMessageBox, QScrollArea)
from PyQt6.QtCore import pyqtSignal, Qt
from common.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class ConfigPanel(QWidget):
    """
    Configuration tab for editing system_config.yaml parameters.
    Allows applying changes at runtime or saving to files.
    """
    # Signal emitted when "Apply" is clicked, passing the new config dictionary
    config_applied = pyqtSignal(dict)

    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.cm = config_manager
        
        # UI Elements
        self.inputs = {}
        self.labels_derived = {}
        
        self._init_ui()
        self.populate_fields()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # ── Profile Selection ─────────────────────────────────────────────
        prof_group = QGroupBox("Profil Kezelés (Telepítési Helyszínek)")
        prof_lay = QHBoxLayout(prof_group)
        
        self.combo_profiles = QComboBox()
        self.combo_profiles.addItems(self.cm.list_profiles())
        prof_lay.addWidget(QLabel("Profil betöltése:"))
        prof_lay.addWidget(self.combo_profiles)
        
        btn_load = QPushButton("📂 Betöltés")
        btn_load.clicked.connect(self._on_load_profile)
        prof_lay.addWidget(btn_load)
        
        prof_lay.addStretch()
        
        self.txt_new_prof = QLineEdit()
        self.txt_new_prof.setPlaceholderText("Új profil neve (pl. palya1)...")
        prof_lay.addWidget(self.txt_new_prof)
        
        btn_save_prof = QPushButton("💾 Mentés új profilként")
        btn_save_prof.clicked.connect(self._on_save_profile)
        prof_lay.addWidget(btn_save_prof)
        
        content_layout.addWidget(prof_group)

        # ── Camera Geometry ───────────────────────────────────────────────
        geom_group = QGroupBox("📐 Kamera Geometria (Sarki Elrendezés)")
        geom_lay = QFormLayout(geom_group)
        
        self.inputs["baseline_mm"] = self._create_line_edit()
        self.inputs["camera_height_mm"] = self._create_line_edit()
        self.inputs["focal_length_mm"] = self._create_line_edit()
        self.inputs["sensor_width_mm"] = self._create_line_edit()
        self.inputs["sensor_height_mm"] = self._create_line_edit()
        
        geom_lay.addRow("Kamerák közötti távolság [Baseline] (mm):", self.inputs["baseline_mm"])
        geom_lay.addRow("Kamerák magassága a talajtól (mm):", self.inputs["camera_height_mm"])
        geom_lay.addRow("Objektív fizikai fókusztávolsága (mm):", self.inputs["focal_length_mm"])
        geom_lay.addRow("Képérzékelő szélessége (mm):", self.inputs["sensor_width_mm"])
        geom_lay.addRow("Képérzékelő magassága (mm):", self.inputs["sensor_height_mm"])
        
        # Derived fields (read-only)
        self.labels_derived["focal_length_px"] = QLabel("---")
        self.labels_derived["focal_length_px"].setStyleSheet("color: #00E5FF; font-weight: bold;")
        geom_lay.addRow("→ Számított Fókusztávolság (px):", self.labels_derived["focal_length_px"])
        
        content_layout.addWidget(geom_group)
        
        # ── Field Geometry ────────────────────────────────────────────────
        field_group = QGroupBox("🏟️ Pálya és Kapu Geometria")
        field_lay = QFormLayout(field_group)
        
        self.inputs["goal_distance_mm"] = self._create_line_edit()
        self.inputs["goal_width_mm"] = self._create_line_edit()
        self.inputs["goal_height_mm"] = self._create_line_edit()
        
        field_lay.addRow("Kapu távolsága a kameráktól [Z tengely] (mm):", self.inputs["goal_distance_mm"])
        field_lay.addRow("Kapu szélessége (mm):", self.inputs["goal_width_mm"])
        field_lay.addRow("Kapu magassága (mm):", self.inputs["goal_height_mm"])
        
        content_layout.addWidget(field_group)
        
        # ── Network ───────────────────────────────────────────────────────
        net_group = QGroupBox("🌐 Hálózat (Robot Vezérlő)")
        net_lay = QFormLayout(net_group)
        
        self.inputs["rpi_ip"] = QLineEdit()
        self.inputs["port"] = self._create_line_edit()
        
        net_lay.addRow("Málna PC (RPi) IP címe:", self.inputs["rpi_ip"])
        net_lay.addRow("UDP Port:", self.inputs["port"])
        
        content_layout.addWidget(net_group)
        
        # Connect text changed signals to recalculate derived params
        for line_edit in self.inputs.values():
            line_edit.textChanged.connect(self._recalc_derived)
            
        content_layout.addStretch()
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        # ── Action Buttons ────────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        
        btn_reset = QPushButton("↺ Visszaállítás")
        btn_reset.clicked.connect(self._on_reload)
        btn_layout.addWidget(btn_reset)
        
        btn_layout.addStretch()
        
        btn_save = QPushButton("💾 Mentés a system_config.yaml fájlba")
        btn_save.clicked.connect(self._on_save_default)
        btn_save.setStyleSheet("background-color: #2E7D32; color: white;")
        btn_layout.addWidget(btn_save)
        
        btn_apply = QPushButton("✅ Alkalmazás (Újraindítás nélkül)")
        btn_apply.clicked.connect(self._on_apply)
        btn_apply.setStyleSheet("background-color: #0277BD; color: white; font-weight: bold;")
        btn_layout.addWidget(btn_apply)
        
        main_layout.addLayout(btn_layout)

    def _create_line_edit(self):
        le = QLineEdit()
        le.setFixedWidth(150)
        return le

    def populate_fields(self):
        """Fills the UI fields from the current config manager state."""
        cfg = self.cm.config
        
        fg = cfg.get("field_geometry", {})
        self.inputs["baseline_mm"].setText(str(fg.get("baseline_mm", "")))
        self.inputs["camera_height_mm"].setText(str(fg.get("camera_height_mm", "")))
        self.inputs["focal_length_mm"].setText(str(fg.get("focal_length_mm", "")))
        self.inputs["sensor_width_mm"].setText(str(fg.get("sensor_width_mm", "")))
        self.inputs["sensor_height_mm"].setText(str(fg.get("sensor_height_mm", "")))
        
        self.inputs["goal_distance_mm"].setText(str(fg.get("goal_distance_mm", "")))
        self.inputs["goal_width_mm"].setText(str(fg.get("goal_width_mm", "")))
        self.inputs["goal_height_mm"].setText(str(fg.get("goal_height_mm", "")))
        
        net = cfg.get("network", {})
        self.inputs["rpi_ip"].setText(str(net.get("rpi_ip", "")))
        self.inputs["port"].setText(str(net.get("port", "")))
        
        self._recalc_derived()

    def _build_config_from_ui(self) -> dict:
        """Constructs a config dictionary from current UI values."""
        # Start with a copy of current config so we don't lose unedited fields
        import copy
        new_cfg = copy.deepcopy(self.cm.config)
        
        if "field_geometry" not in new_cfg:
            new_cfg["field_geometry"] = {}
        if "network" not in new_cfg:
            new_cfg["network"] = {}
            
        try:
            fg = new_cfg["field_geometry"]
            fg["baseline_mm"] = float(self.inputs["baseline_mm"].text() or 0)
            fg["camera_height_mm"] = float(self.inputs["camera_height_mm"].text() or 0)
            fg["focal_length_mm"] = float(self.inputs["focal_length_mm"].text() or 0)
            fg["sensor_width_mm"] = float(self.inputs["sensor_width_mm"].text() or 0)
            fg["sensor_height_mm"] = float(self.inputs["sensor_height_mm"].text() or 0)
            fg["goal_distance_mm"] = float(self.inputs["goal_distance_mm"].text() or 0)
            fg["goal_width_mm"] = float(self.inputs["goal_width_mm"].text() or 0)
            fg["goal_height_mm"] = float(self.inputs["goal_height_mm"].text() or 0)
            
            # Keep trajectory z-goal synced with goal_distance
            if "trajectory" not in new_cfg:
                new_cfg["trajectory"] = {}
            new_cfg["trajectory"]["goal_z_mm"] = fg["goal_distance_mm"]
            
            net = new_cfg["network"]
            net["rpi_ip"] = self.inputs["rpi_ip"].text()
            net["port"] = int(self.inputs["port"].text() or 0)
            
            # Recalculate derived params
            new_cfg = self.cm.derive_stereo_params(new_cfg)
            return new_cfg
            
        except ValueError as e:
            QMessageBox.warning(self, "Hiba", f"Érvénytelen számformátum: {e}")
            return None

    def _recalc_derived(self):
        """Updates the read-only derived labels."""
        cfg = self._build_config_from_ui()
        if cfg:
            f_px = cfg.get("stereo", {}).get("focal_length_px", "---")
            self.labels_derived["focal_length_px"].setText(f"{f_px} px")

    def _on_reload(self):
        self.cm.load()
        self.populate_fields()
        
    def _on_save_default(self):
        cfg = self._build_config_from_ui()
        if not cfg: return
        
        errors = self.cm.validate(cfg)
        if errors:
            QMessageBox.warning(self, "Validációs Hiba", "\\n".join(errors))
            return
            
        self.cm.config = cfg
        self.cm.save(cfg)
        QMessageBox.information(self, "Mentve", "A konfiguráció sikeresen elmentve a system_config.yaml fájlba.")
        
    def _on_apply(self):
        cfg = self._build_config_from_ui()
        if not cfg: return
        
        errors = self.cm.validate(cfg)
        if errors:
            QMessageBox.warning(self, "Validációs Hiba", "\\n".join(errors))
            return
            
        self.cm.config = cfg
        self.config_applied.emit(cfg)
        QMessageBox.information(self, "Alkalmazva", "A beállítások azonnal érvénybe léptek a futó rendszerben.")

    def _on_load_profile(self):
        prof_name = self.combo_profiles.currentText()
        if not prof_name: return
        
        try:
            self.cm.load_profile(prof_name)
            self.populate_fields()
        except Exception as e:
            QMessageBox.critical(self, "Hiba", f"Nem sikerült betölteni a profilt: {e}")

    def _on_save_profile(self):
        prof_name = self.txt_new_prof.text().strip()
        if not prof_name:
            QMessageBox.warning(self, "Név megadása kötelező", "Kérlek adj meg egy nevet az új profilnak.")
            return
            
        cfg = self._build_config_from_ui()
        if not cfg: return
        
        try:
            self.cm.save_profile(cfg, prof_name)
            self.combo_profiles.clear()
            self.combo_profiles.addItems(self.cm.list_profiles())
            self.combo_profiles.setCurrentText(prof_name)
            self.txt_new_prof.clear()
            QMessageBox.information(self, "Mentve", f"A '{prof_name}' profil sikeresen elmentve.")
        except Exception as e:
            QMessageBox.critical(self, "Hiba", f"Nem sikerült elmenteni a profilt: {e}")
