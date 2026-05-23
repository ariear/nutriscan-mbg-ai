import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyNutritionClassifier:
    STATUS_LABELS = {
        (80, 101): "seimbang",
        (60,  80): "cukup_seimbang",
        (40,  60): "kurang_seimbang",
        (20,  40): "tidak_seimbang",
        (0,   20): "sangat_tidak_seimbang",
    }

    def __init__(self) -> None:
        self._build_system()

    def _build_system(self) -> None:
        self.karbo   = ctrl.Antecedent(np.arange(0, 101, 1), "karbo")
        self.protein = ctrl.Antecedent(np.arange(0, 101, 1), "protein")
        self.serat   = ctrl.Antecedent(np.arange(0, 101, 1), "serat")
        self.susu    = ctrl.Antecedent(np.arange(0, 101, 1), "susu")
        self.score   = ctrl.Consequent(np.arange(0, 101, 1), "score")

        self.karbo["rendah"] = fuzz.trimf(self.karbo.universe, [0,   0,  40])
        self.karbo["sedang"] = fuzz.trimf(self.karbo.universe, [30,  50, 70])
        self.karbo["tinggi"] = fuzz.trimf(self.karbo.universe, [60, 100, 100])
        self.protein["kurang"] = fuzz.trimf(self.protein.universe, [0,   0,  20])
        self.protein["cukup"]  = fuzz.trimf(self.protein.universe, [15,  25, 35])
        self.protein["lebih"]  = fuzz.trimf(self.protein.universe, [30, 100, 100])
        self.serat["kurang"] = fuzz.trimf(self.serat.universe, [0,   0,  20])
        self.serat["cukup"]  = fuzz.trimf(self.serat.universe, [15,  25, 40])
        self.serat["lebih"]  = fuzz.trimf(self.serat.universe, [35, 100, 100])
        self.susu["tidak_ada"] = fuzz.trimf(self.susu.universe, [0,   0,  5])
        self.susu["ada"]       = fuzz.trimf(self.susu.universe, [3,  100, 100])
        self.score["sangat_buruk"] = fuzz.trimf(self.score.universe, [0,   0,  20])
        self.score["buruk"]        = fuzz.trimf(self.score.universe, [10,  25, 40])
        self.score["sedang"]       = fuzz.trimf(self.score.universe, [30,  50, 70])
        self.score["baik"]         = fuzz.trimf(self.score.universe, [60,  75, 90])
        self.score["sangat_baik"]  = fuzz.trimf(self.score.universe, [80, 100, 100])

        rules = [
            ctrl.Rule(
                self.karbo["sedang"] & self.protein["cukup"] &
                self.serat["cukup"]  & self.susu["ada"],
                self.score["sangat_baik"],
            ),
            ctrl.Rule(
                self.karbo["sedang"] & self.protein["cukup"] &
                self.serat["cukup"]  & self.susu["tidak_ada"],
                self.score["baik"],
            ),

            ctrl.Rule(self.karbo["tinggi"], self.score["buruk"]),
            ctrl.Rule(
                self.karbo["tinggi"] & self.protein["kurang"],
                self.score["sangat_buruk"],
            ),

            ctrl.Rule(self.protein["kurang"], self.score["buruk"]),
            ctrl.Rule(
                self.protein["kurang"] & self.serat["kurang"],
                self.score["sangat_buruk"],
            ),

            ctrl.Rule(self.serat["kurang"], self.score["buruk"]),

            ctrl.Rule(
                self.susu["tidak_ada"] & self.karbo["sedang"] & self.protein["cukup"],
                self.score["sedang"],
            ),

            ctrl.Rule(
                self.karbo["rendah"] & self.protein["cukup"],
                self.score["sedang"],
            ),
            ctrl.Rule(
                self.karbo["sedang"] & self.protein["cukup"] & self.serat["kurang"],
                self.score["sedang"],
            ),
        ]

        self.ctrl_system = ctrl.ControlSystem(rules)
        self.simulation  = ctrl.ControlSystemSimulation(self.ctrl_system)

    def classify(self, nutrisi: dict) -> dict:
        self.simulation.input["karbo"]   = float(np.clip(nutrisi.get("karbo",   0), 0, 100))
        self.simulation.input["protein"] = float(np.clip(nutrisi.get("protein", 0), 0, 100))
        self.simulation.input["serat"]   = float(np.clip(nutrisi.get("serat",   0), 0, 100))
        self.simulation.input["susu"]    = float(np.clip(nutrisi.get("susu",    0), 0, 100))

        try:
            self.simulation.compute()
            healthy_score = round(float(self.simulation.output["score"]), 1)
        except Exception:
            healthy_score = 50.0

        status = "tidak_seimbang"
        for (low, high), lbl in self.STATUS_LABELS.items():
            if low <= healthy_score < high:
                status = lbl
                break

        detail = self._rule_based_detail(nutrisi)

        return {
            "healthy_score": healthy_score,
            "status"       : status,
            "detail"       : detail,
            "nutrisi"      : nutrisi,
        }

    def _rule_based_detail(self, nutrisi: dict) -> str:
        k = nutrisi.get("karbo",   0)
        p = nutrisi.get("protein", 0)
        s = nutrisi.get("serat",   0)
        u = nutrisi.get("susu",    0)

        if k > 60:
            return "tinggi_karbo"
        if p < 15:
            if s < 15:
                return "protein_dan_serat_kurang"
            return "protein_kurang"
        if s < 15:
            return "serat_kurang"
        if u == 0:
            return "tanpa_susu"
        if 35 <= k <= 65 and p >= 15 and s >= 15 and u > 0:
            return "seimbang"
        return "kurang_seimbang"

