# ============================================
# Compute area of Trial 94 seawall cross section
# ============================================

# Trial 94 polygon coordinates (meters)
points = [
    (460.000, -0.700),
    (460.000,  0.700),
    (460.556,  0.881),
    (461.111,  1.323),
    (461.667,  1.634),
    (462.222,  1.850),
    (462.778,  1.888),
    (463.333,  2.428),
    (463.889,  2.838),
    (464.444,  3.217),
    (465.000,  3.377),
    (465.000, -3.000),
    (460.000, -0.700)
]

# Shoelace formula
def polygon_area(coords):
    area = 0.0
    n = len(coords)
    
    for i in range(n - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        area += (x1 * y2) - (x2 * y1)
    
    return abs(area) / 2.0

# Compute area
area = polygon_area(points)

print("Trial 94 seawall cross-sectional area:")
print(f"Area = {area:.6f} m^2")

# Compare to 6 m^2 reference
reference = 6.0
print(f"\nComparison to 6 m^2:")
print(f"Ratio = {area/reference:.6f}")
