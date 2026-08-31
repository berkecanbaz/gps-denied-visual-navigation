"""
GPS-denied drone localization.

Matches a downward-facing drone image against a georeferenced satellite map
and estimates the drone's GPS position from the resulting homography.

Pipeline:
    SIFT features -> Brute Force matching (Lowe ratio test) -> RANSAC homography
    -> perspective transform of the image center -> bilinear pixel-to-GPS mapping
"""

import cv2
import numpy as np


# --- Configuration -----------------------------------------------------------

DRONE_IMAGE = "data/drone.jpg"
SATELLITE_IMAGE = "data/uydu2.jpg"

DRONE_MAX_SIZE = 800
SATELLITE_MAX_SIZE = 1400

SIFT_FEATURES = 5000
LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 5.0
MIN_MATCHES = 10


def dms_to_dd(degrees, minutes, seconds):
    """Convert degrees-minutes-seconds to decimal degrees."""
    return degrees + minutes / 60 + seconds / 3600


# Corner coordinates of the satellite map, read from Google Earth.
# Order: top-left, top-right, bottom-left, bottom-right.
SATELLITE_CORNERS = {
    "top_left":     (dms_to_dd(40, 59, 49.24), dms_to_dd(28, 42, 44.82)),
    "top_right":    (dms_to_dd(41,  0,  5.38), dms_to_dd(28, 43, 22.94)),
    "bottom_left":  (dms_to_dd(40, 59, 32.93), dms_to_dd(28, 42, 56.01)),
    "bottom_right": (dms_to_dd(40, 59, 48.17), dms_to_dd(28, 43, 34.31)),
}


# --- Helpers -----------------------------------------------------------------

def resize(image, max_size):
    """Scale an image down so its longest side is at most max_size."""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def find_matches(drone_gray, satellite_gray):
    """Detect SIFT features in both images and return the filtered matches."""
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    kp_drone, des_drone = sift.detectAndCompute(drone_gray, None)
    kp_sat, des_sat = sift.detectAndCompute(satellite_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(des_drone, des_sat, k=2)

    # Lowe's ratio test: keep a match only if the closest neighbour is
    # clearly better than the second closest.
    good = [m for m, n in raw_matches if m.distance < LOWE_RATIO * n.distance]

    return kp_drone, kp_sat, good


def estimate_homography(kp_drone, kp_sat, matches):
    """Fit a homography from drone image coordinates to satellite coordinates."""
    src = np.float32([kp_drone[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_sat[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESHOLD)
    return H, int(mask.sum())


def pixel_to_gps(pixel_x, pixel_y, map_width, map_height, corners):
    """
    Convert a pixel position on the satellite map to latitude and longitude
    by bilinearly interpolating between the map's four corner coordinates.
    """
    u = pixel_x / map_width
    v = pixel_y / map_height

    tl_lat, tl_lon = corners["top_left"]
    tr_lat, tr_lon = corners["top_right"]
    bl_lat, bl_lon = corners["bottom_left"]
    br_lat, br_lon = corners["bottom_right"]

    # Interpolate along the top and bottom edges, then between them.
    top_lat = tl_lat + u * (tr_lat - tl_lat)
    top_lon = tl_lon + u * (tr_lon - tl_lon)
    bottom_lat = bl_lat + u * (br_lat - bl_lat)
    bottom_lon = bl_lon + u * (br_lon - bl_lon)

    lat = top_lat + v * (bottom_lat - top_lat)
    lon = top_lon + v * (bottom_lon - top_lon)

    return lat, lon


def draw_result(drone, satellite, H, center_x, center_y, lat, lon):
    """Draw the drone's footprint and estimated position on the satellite map."""
    h, w = drone.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H)

    overlay = satellite.copy()
    cv2.polylines(overlay, [np.int32(projected)], True, (0, 255, 0), 3)

    x, y = int(center_x), int(center_y)
    cv2.circle(overlay, (x, y), 20, (0, 0, 255), 4)
    cv2.putText(overlay, f"{lat:.5f}, {lon:.5f}", (x + 25, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(overlay, "DRONE", (x + 25, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    scale = satellite.shape[0] / drone.shape[0]
    drone_scaled = cv2.resize(drone, None, fx=scale, fy=scale)

    return resize(np.hstack((drone_scaled, overlay)), 1800)


# --- Main --------------------------------------------------------------------

def main():
    drone = cv2.imread(DRONE_IMAGE)
    satellite = cv2.imread(SATELLITE_IMAGE)

    if drone is None or satellite is None:
        print(f"Could not read {DRONE_IMAGE} or {SATELLITE_IMAGE}")
        return

    drone = resize(drone, DRONE_MAX_SIZE)
    satellite = resize(satellite, SATELLITE_MAX_SIZE)

    drone_gray = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)
    satellite_gray = cv2.cvtColor(satellite, cv2.COLOR_BGR2GRAY)

    kp_drone, kp_sat, matches = find_matches(drone_gray, satellite_gray)
    print(f"Matches passing the ratio test: {len(matches)}")

    if len(matches) < MIN_MATCHES:
        print("Not enough matches to estimate a homography.")
        return

    H, inliers = estimate_homography(kp_drone, kp_sat, matches)
    print(f"RANSAC inliers: {inliers}/{len(matches)} "
          f"({100 * inliers / len(matches):.0f}%)")

    # Project the drone image center onto the satellite map.
    h, w = drone.shape[:2]
    center = np.array([[w / 2, h / 2]], dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(center, H)
    pixel_x = float(projected[0][0][0])
    pixel_y = float(projected[0][0][1])

    map_h, map_w = satellite.shape[:2]
    print(f"Position on the satellite map: ({pixel_x:.1f}, {pixel_y:.1f}) px")

    lat, lon = pixel_to_gps(pixel_x, pixel_y, map_w, map_h, SATELLITE_CORNERS)

    print(f"\nEstimated GPS position: {lat:.6f}, {lon:.6f}")
    print(f"https://maps.google.com/?q={lat:.6f},{lon:.6f}")

    result = draw_result(drone, satellite, H, pixel_x, pixel_y, lat, lon)
    cv2.imshow("Drone GPS position estimate", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
