"""Fixed baseline/v1 parameters, chosen by the architect without public-GT tuning."""

from games import GAME, GAMES

# Params
FPS = GAMES[GAME]["fps"]  # Tracking and automatic tags share the tactical video's clock.
PITCH = GAMES[GAME]["pitch"]  # Normalised coordinates must use the measured pitch in metres.
HOME, AWAY, DEAD = 0, 1, 2  # State order also resolves equal evidence deterministically.
NOBODY = -1  # Missing ball/player observations provide no baseline possession evidence.
CACHE_VERSION = 1  # Invalidate stored coordinates when the tracking representation changes.
DEFAULT_WINDOW = 50  # Published baseline smooths tracking flicker over +/- two seconds.

MAX_GAP_FRAMES = 12  # Bridge only roughly half a second of ball occlusion.
MAX_INTERPOLATION_SPEED_MPS = 40.0  # Longer implied jumps are implausible ball motion.
SPEED_MEDIAN_FRAMES = 5  # A short median filter suppresses tracking jitter in ball speed.
OUT_MARGIN_M = 0.5  # Allow half a metre of line-location noise before declaring out.

CONTROL_RADIUS_M = 1.5  # A player must be within touching distance to control the ball.
CONTROL_SPEED_MPS = 7.0  # A fast-moving ball is in flight rather than controlled.
CONTEST_MARGIN_M = 1.0  # Similar opponent distances weaken claims of clear control.
W_CONTROL = 2.0  # Close, slow ball observations are the strongest team-control evidence.
INTERPOLATED_CONTROL_WEIGHT = 0.5  # Inferred ball locations are less reliable than observations.
W_TAG_ATTACK = 0.3  # Automatic attacking phases supply only a weak team prior.
W_TAG_DEFEND = 0.2  # Defensive phases weakly imply the other side has possession.
TAG_CAP = 0.6  # Correlated overlapping phases must not overwhelm direct ball evidence.

W_OUT = 3.0  # A directly observed ball beyond the lines strongly implies dead play.
STILL_WINDOW_S = 2.0  # Average collective motion over +/- two seconds to detect stoppages.
STILL_SPEED_MPS = 1.2  # A mostly walking or stationary team suggests a stoppage.
W_STILL = 1.0  # Sustained collective stillness is evidence for dead play.
W_STILL_NO_BALL = 0.5  # Stillness is stronger evidence when the ball is absent throughout.
RESTART_LEAD_S = 3.0  # The few seconds before an automatic restart tag are likely dead.
W_RESTART_DEAD = 1.0  # Restart tags weakly identify the preceding stoppage.
RESTART_ALIVE_S = 2.0  # A known restarting team gets a brief possession prior after restart.
W_RESTART_ALIVE = 0.5  # Restart possession is useful but the automatic tag can be wrong.
W_PREKICKOFF = 10.0  # The graded half before its tagged kickoff is strongly dead play.

P_SWITCH = 8.0  # Team turnovers need sustained evidence, not a single noisy observation.
P_TO_DEAD = 6.0  # Ending live play needs stronger evidence than simply staying alive.
P_FROM_DEAD = 4.0  # Restarting play requires evidence but is easier than inventing a stop.
EMISSION_BASELINE = 0.0  # Equal log-scores are uniform evidence over H, A and D.

ATTACK_CODES = frozenset({  # These automatic phases describe the side attacking with the ball.
    "build up", "progression", "creating chances", "ball in final third",
    "ball in the box", "players in the box", "attacking transition", "long ball",
})
DEFEND_CODES = frozenset({  # These phases describe the defending side, hence opposite possession.
    "defending in attacking third", "defending in middle third",
    "defending in defensive third", "defensive transition",
})
TEAM_NAMES = {"barcelona": HOME, "atletic club": AWAY}  # N/A supplies no team identity.
HALF_NAMES = {"1st half": 1, "2nd half": 2}  # Match automatic kickoff tags to graded halves.
SET_PIECES_CODE = "set pieces"  # Restart timing comes from automatic set-piece intervals.
KICKOFF_START_TYPE = "kick off (start)"  # Only half-start kickoffs create a pre-half dead prior.
