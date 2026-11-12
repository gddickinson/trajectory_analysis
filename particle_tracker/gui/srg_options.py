"""SrgOptionsWidget: sRg mobility options (preset, definition, threshold, minimum track length) for the desktop GUI.

All values and texts come from analysis/mobility.py (presets, definitions, defaults); nothing is hard-coded here.
Choosing a preset fills the threshold and minimum length; editing either switches the preset to "custom".
Switching the definition converts the threshold so the same cut is kept.
"""
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QSpinBox, QWidget

from ..analysis import mobility

DEFINITION_LABELS = {
    "golan_sherman": "√(π/2)·Rg / mean step (Golan & Sherman; recommended)",
    "rg_over_step": "Rg / mean step (pathak_lab scripts' radius_gyration_scaled)",
}


class SrgOptionsWidget(QWidget):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = False
        self._definition = mobility.DEFAULT_DEFINITION
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)

        self.preset_combo = QComboBox()
        for key, p in mobility.PRESETS.items():
            self.preset_combo.addItem(p["label"], key)
        self.preset_combo.addItem("Custom", mobility.CUSTOM)
        form.addRow("sRg preset:", self.preset_combo)

        self.definition_combo = QComboBox()
        for key in mobility.SRG_DEFINITIONS:
            self.definition_combo.addItem(DEFINITION_LABELS.get(key, key), key)
            self.definition_combo.setItemData(self.definition_combo.count() - 1,
                                              mobility.SRG_DEFINITIONS[key][1], 3)  # tooltip role
        form.addRow("sRg definition:", self.definition_combo)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 100.0)
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setSingleStep(0.05)
        form.addRow("Mobile above (sRg):", self.threshold_spin)

        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(0, 100000)
        self.min_length_spin.setSuffix(" points")
        self.min_length_spin.setToolTip("Tracks with fewer points are reported as 'too short', not classified "
                                        "(0 = classify every track, the legacy behaviour).")
        form.addRow("Classify tracks with ≥:", self.min_length_spin)

        self.basis_label = QLabel()
        self.basis_label.setWordWrap(True)
        self.basis_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        form.addRow("", self.basis_label)

        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        self.definition_combo.currentIndexChanged.connect(self._on_definition)
        self.threshold_spin.valueChanged.connect(self._on_edit)
        self.min_length_spin.valueChanged.connect(self._on_edit)
        self.reset()

    # --- public API -------------------------------------------------------------------------------------------
    def values(self) -> dict:
        preset, definition = self.preset_combo.currentData(), self.definition_combo.currentData()
        if preset != mobility.CUSTOM:   # the preset's exact threshold (the spin box shows 4 decimals)
            return mobility.resolve({"srg_preset": preset, "srg_definition": definition})
        return {"srg_preset": preset, "srg_definition": definition, "mobility_threshold": self.threshold_spin.value(),
                "min_classify_length": self.min_length_spin.value()}

    def set_values(self, params: dict):
        """Apply any of srg_preset / srg_definition / mobility_threshold / min_classify_length (resolved)."""
        r = mobility.resolve({k: params[k] for k in mobility.MOBILITY_DEFAULTS if k in params})
        if r["srg_preset"] != mobility.CUSTOM:
            ref = mobility.resolve({"srg_preset": r["srg_preset"], "srg_definition": r["srg_definition"]})
            if (abs(ref["mobility_threshold"] - r["mobility_threshold"]) > 1e-9
                    or ref["min_classify_length"] != r["min_classify_length"]):
                r["srg_preset"] = mobility.CUSTOM
        self._busy = True
        try:
            self._select(self.preset_combo, r["srg_preset"])
            self._select(self.definition_combo, r["srg_definition"])
            self._definition = r["srg_definition"]
            self.threshold_spin.setValue(r["mobility_threshold"])
            self.min_length_spin.setValue(r["min_classify_length"])
        finally:
            self._busy = False
        self._show_basis()
        self.valueChanged.emit()

    def reset(self):
        self.set_values({})

    # --- handlers ---------------------------------------------------------------------------------------------
    def _on_preset(self):
        if self._busy:
            return
        key = self.preset_combo.currentData()
        if key != mobility.CUSTOM:
            self.set_values({"srg_preset": key, "srg_definition": self.definition_combo.currentData()})
        else:
            self._show_basis()
            self.valueChanged.emit()

    def _on_definition(self):
        if self._busy:
            return
        new = self.definition_combo.currentData()
        self._busy = True
        try:
            self.threshold_spin.setValue(mobility.convert_threshold(self.threshold_spin.value(), self._definition, new))
        finally:
            self._busy = False
        self._definition = new
        self.valueChanged.emit()

    def _on_edit(self):
        if self._busy:
            return
        key = self.preset_combo.currentData()
        if key != mobility.CUSTOM:
            p = mobility.PRESETS[key]
            expected = mobility.convert_threshold(p["mobility_threshold"], mobility.DEFAULT_DEFINITION,
                                                  self._definition)
            if (abs(self.threshold_spin.value() - expected) > 5e-5
                    or self.min_length_spin.value() != p["min_classify_length"]):
                self._busy = True
                self._select(self.preset_combo, mobility.CUSTOM)
                self._busy = False
                self._show_basis()
        self.valueChanged.emit()

    def _show_basis(self):
        key = self.preset_combo.currentData()
        self.basis_label.setText(mobility.PRESETS[key]["basis"] if key in mobility.PRESETS else
                                 "Custom values. Calibration: docs/srg_calibration.md.")

    @staticmethod
    def _select(combo, key):
        i = combo.findData(key)
        if i >= 0:
            combo.setCurrentIndex(i)
