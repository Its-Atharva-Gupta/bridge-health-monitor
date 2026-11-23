"""Anomaly detectors that learn ONLY from healthy test cycles.

Common steps for every detector:
  1. Weather correction: learn (from healthy cycles) how temperature and humidity
     shift each feature, and subtract that effect.
  2. Standardise each feature (mean 0, spread 1) so no feature dominates by units.
  3. Score: how unusual is this cycle compared with healthy?
  4. Alarm threshold: chosen so ~99% of healthy cycles score below it. Measured
     with leave-one-out (each healthy cycle is scored by a model that never saw
     it), otherwise the threshold would be unrealistically tight.

Detectors:
  MahalanobisDetector  - main model. Distance from "normal" that accounts for how
                         the features normally move together.
  IsolationForestDetector - a different ML method, for comparison.
  SagThresholdDetector - the simple non-ML approach: "is the sag unusual?"
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import config as C


class _Base:
    name = "base"

    def __init__(self, features=None, env=None, quantile=0.99):
        self.features = list(features or C.FEATURES)
        self.env = list(env or C.ENV_COLUMNS)
        self.quantile = quantile

    # ---- weather correction + scaling (shared) ----
    def _prep_fit(self, df):
        X = df[self.features].to_numpy(float)
        E = df[self.env].to_numpy(float)
        env_scaler = StandardScaler().fit(E)
        reg = Ridge(alpha=1.0).fit(env_scaler.transform(E), X)
        R = X - reg.predict(env_scaler.transform(E)).reshape(X.shape)
        scaler = StandardScaler().fit(R)
        return dict(env_scaler=env_scaler, reg=reg, scaler=scaler), scaler.transform(R)

    @staticmethod
    def _prep_apply(prep, df, features, env):
        X = df[features].to_numpy(float)
        E = df[env].to_numpy(float)
        R = X - prep["reg"].predict(prep["env_scaler"].transform(E)).reshape(X.shape)
        return prep["scaler"].transform(R)

    # ---- detector-specific ----
    def _core_fit(self, Z):
        raise NotImplementedError

    def _core_score(self, core, Z):
        raise NotImplementedError

    # ---- public API ----
    def _fit_once(self, df):
        prep, Z = self._prep_fit(df)
        return prep, self._core_fit(Z)

    def _score_with(self, fitted, df):
        prep, core = fitted
        return self._core_score(core, self._prep_apply(prep, df, self.features, self.env))

    def fit(self, healthy_df):
        healthy_df = healthy_df.reset_index(drop=True)
        # drop features this rig can't measure (missing sensor) or that never vary
        usable = [f for f in self.features
                  if healthy_df[f].notna().all() and healthy_df[f].std() > 0]
        self.dropped_ = [f for f in self.features if f not in usable]
        self.features = usable
        self.fitted_ = self._fit_once(healthy_df)
        loo = np.array([
            self._score_with(self._fit_once(healthy_df.drop(index=i)), healthy_df.iloc[[i]])[0]
            for i in range(len(healthy_df))
        ])
        self.loo_scores_ = loo
        self.threshold_ = float(np.quantile(loo, self.quantile))
        self.typical_ = float(np.median(loo))
        return self

    def score(self, df):
        return self._score_with(self.fitted_, df)

    def is_anomaly(self, df):
        return self.score(df) > self.threshold_

    # ---- friendly outputs for the dashboard / voice ----
    def health(self, df):
        """0-100: 100 = typical healthy, 60 = at the alarm threshold, 0 = far beyond."""
        s = self.score(df)
        xp = [self.typical_, self.threshold_, 2 * self.threshold_, 4 * self.threshold_]
        return np.interp(s, xp, [100, 60, 30, 0])

    def status(self, df):
        s = self.score(df)
        return np.where(s < self.threshold_, "Healthy",
                        np.where(s < 2 * self.threshold_, "Warning", "Damaged"))

    def explain(self, row_df, top=3):
        """Which features are most unusual (in healthy standard deviations)."""
        prep, _ = self.fitted_
        z = self._prep_apply(prep, row_df, self.features, self.env)[0]
        order = np.argsort(-np.abs(z))[:top]
        return [(self.features[i], float(z[i])) for i in order]


class MahalanobisDetector(_Base):
    name = "Mahalanobis"

    def _core_fit(self, Z):
        # Ledoit-Wolf: a covariance estimate that stays stable with few samples
        lw = LedoitWolf().fit(Z)
        return dict(mean=lw.location_, prec=np.linalg.inv(lw.covariance_))

    def _core_score(self, core, Z):
        D = Z - core["mean"]
        return np.sqrt(np.einsum("ij,jk,ik->i", D, core["prec"], D))


class IsolationForestDetector(_Base):
    name = "Isolation Forest"

    def _core_fit(self, Z):
        return IsolationForest(n_estimators=200, random_state=0).fit(Z)

    def _core_score(self, core, Z):
        return -core.score_samples(Z)          # higher = more unusual


class SagThresholdDetector(_Base):
    """Non-ML baseline: only looks at sag at max load (weather-corrected, for fairness)."""
    name = "Sag threshold"

    def __init__(self, **kw):
        kw.setdefault("features", ["sag_at_max_mm"])
        super().__init__(**kw)

    def _core_fit(self, Z):
        return None

    def _core_score(self, core, Z):
        return np.abs(Z[:, 0])


DETECTORS = [MahalanobisDetector, IsolationForestDetector, SagThresholdDetector]
