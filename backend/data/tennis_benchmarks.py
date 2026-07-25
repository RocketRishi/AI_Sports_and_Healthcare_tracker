"""
Reference biomechanics norms for the tennis serve, compiled from
published elite-player studies. These are the numbers tennis_analysis
compares a measured swing against.

Angle convention: every *_flexion_deg / *_lean_deg value here is a
"deviation from straight/upright" angle (0 = straight or upright),
matching core.angle_engine.calculate_flexion_angle /
calculate_lean_angle. This is NOT the same as the raw interior angle
returned by calculate_angle().

Sources:
  - Kovacs & Ellenbecker (2011), "An 8-Stage Model for Evaluating the
    Tennis Serve" — MER shoulder/elbow/wrist angles.
  - Systematic review & meta-analysis of 27 serve biomechanics studies
    (2024) — trophy/MER/impact pooled means and SDs.
  - Bruce Elliott, Olympic-cohort serve kinetics study — peak joint
    torques and the angular velocity summation sequence.
  - "Injury Risk Factors of the Tennis Serve: A Systematic Review"
    (2025) — kinetic-chain injury-risk relationships.

All ranges are elite-adult-player norms. They are guidance for
comparison, not a diagnostic or medical threshold.
"""

TROPHY_POSITION = {
    "front_knee_flexion_deg": {
        "mean": 64.5, "sd": 9.7,
        "range": (47, 81),
        "source": "2024 meta-analysis (27 studies)",
    },
    "trunk_lean_deg": {
        "mean": 25.0, "sd": 7.1,
        "source": "2024 meta-analysis (27 studies)",
    },
    "elbow_flexion_deg": {
        "range": (85, 107),
        "source": "recent narrative review",
    },
}

MAX_EXTERNAL_ROTATION = {
    # True glenohumeral external rotation needs 3D/marker-based
    # capture; a monocular 2D camera cannot measure it directly. The
    # angle actually computed here (elbow flexion + shoulder
    # abduction at the "racket drop" frame) is used as a 2D-observable
    # proxy for this phase, not a measurement of the 172 degree figure
    # itself. Both are kept for reference/reporting.
    "shoulder_external_rotation_deg": {
        "mean": 172, "sd": 12,
        "source": "Kovacs & Ellenbecker 2011, 8-stage model",
        "measurable_from_2d_video": False,
    },
    "shoulder_external_rotation_meta_deg": {
        "mean": 130.1, "sd": 26.5,
        "source": "2024 meta-analysis (27 studies)",
        "measurable_from_2d_video": False,
    },
    "shoulder_abduction_deg": {
        "mean": 101, "sd": 13,
        "source": "Kovacs & Ellenbecker 2011",
    },
    "elbow_flexion_deg": {
        "mean": 104, "sd": 12,
        "range": (104, 112),
        "source": "Kovacs & Ellenbecker 2011 / recent review",
    },
    "wrist_extension_deg": {
        "mean": 66, "sd": 19,
        "source": "Kovacs & Ellenbecker 2011",
        "measurable_from_2d_video": False,
    },
}

IMPACT_POSITION = {
    "trunk_lean_deg": {
        "mean": 48,
        "source": "Kovacs & Ellenbecker 2011",
    },
    "shoulder_abduction_deg": {
        "range": (101, 115),
        "source": "pooled elite-player studies",
    },
    "shoulder_elevation_deg": {
        "mean": 110.7, "sd": 16.9,
        "source": "2024 meta-analysis (27 studies)",
    },
    "elbow_flexion_deg": {
        "mean": 30.1, "sd": 15.9,
        "range": (27, 44),
        "source": "2024 meta-analysis (27 studies) / recent review",
    },
    "knee_flexion_deg": {
        "range": (6, 29),
        "source": "pooled elite-player studies",
        "note": "legs have already extended by contact",
    },
}

# Chronological proximal-to-distal summation of peak segment angular
# velocities during the acceleration phase. Cannot be computed from a
# 2D single-camera pose stream (needs 3D segment orientation over
# time); kept here purely as reference context surfaced in feedback.
PEAK_ANGULAR_VELOCITY_SEQUENCE = [
    {"segment": "trunk_tilt", "peak_deg_per_s": 280},
    {"segment": "pelvis_rotation", "peak_deg_per_s": 440},
    {"segment": "upper_torso_rotation", "peak_deg_per_s": 870},
    {"segment": "elbow_extension", "peak_deg_per_s": 1510},
    {"segment": "wrist_flexion", "peak_deg_per_s": 1950},
    {
        "segment": "shoulder_internal_rotation",
        "peak_deg_per_s_male": 2420,
        "peak_deg_per_s_female": 1370,
    },
]

# Peak joint loads from healthy elite players, for context only —
# torque cannot be derived from video (needs force plates/EMG). Useful
# for explaining *why* a form deviation matters, not as a computed
# per-user metric.
PEAK_JOINT_TORQUE = {
    "shoulder_internal_rotation_torque_nm": {"male": 64.9, "female": 37.5},
    "elbow_varus_torque_nm": {"male": 67.6, "female": 41.3},
    "source": "Elliott, Olympic-cohort serve kinetics study",
    "measurable_from_2d_video": False,
}

# Rule text used by tennis_analysis to explain *why* a measured
# deviation matters, keyed by the same issue codes it emits.
INJURY_RISK_NOTES = {
    "low_trophy_knee_flexion": (
        "Shallow knee bend in the loading phase reduces leg-driven "
        "power, forcing the shoulder and elbow to generate more of "
        "the serve's force. This is one of the more consistent "
        "injury-risk findings in the literature: more knee flexion "
        "at the trophy position correlates with lower shoulder and "
        "elbow internal-rotation torque."
    ),
    "excessive_trophy_trunk_lean": (
        "Trunk lean well beyond the elite ~25 degree norm at the "
        "trophy position can compress the lumbar spine as the back "
        "extends further than the elite range."
    ),
    "insufficient_trophy_trunk_lean": (
        "Too little trunk lean at the trophy position under-stores "
        "elastic energy in the trunk, shifting more of the load onto "
        "the arm to generate racket speed."
    ),
    "shallow_mer_elbow_flexion": (
        "Insufficient elbow flexion during the cocking phase shortens "
        "the racket-drop and reduces the efficiency of the kinetic "
        "chain, which increases the demand placed on the shoulder."
    ),
    "high_shoulder_abduction": (
        "Shoulder abduction above the elite range is associated with "
        "subacromial impingement and superior labrum stress."
    ),
    "low_shoulder_abduction": (
        "Shoulder abduction below the elite range typically comes "
        "with reduced racket-head speed at contact."
    ),
    "early_elbow_extension": (
        "Straightening the elbow before or right at impact instead of "
        "through it skips the final, most efficient stage of the "
        "kinetic chain and increases stress on the shoulder and the "
        "elbow's medial (UCL) structures."
    ),
    "elbow_hyperextension": (
        "The elbow is locking out beyond a straight line. Repeated "
        "hyperextension loads the joint capsule directly rather than "
        "through muscle, which is a mechanism for elbow injury."
    ),
    "low_impact_knee_flexion": (
        "Elite servers still have slight knee flexion at contact; a "
        "fully locked leg at impact means the legs finished "
        "contributing power too early, again shifting load upward "
        "to the shoulder."
    ),
}
