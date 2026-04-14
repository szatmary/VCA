#!/usr/bin/env bash
# Interleaved stable-vs-highway benchmark for the Highway SIMD port.
#
# Runs VCA on ~3 minutes of Tears of Steel YUV, alternating branches so
# thermal state is comparable between matched stable/highway runs. This
# avoids the systematic bias you get from running all of one branch then
# all of the other, where the second branch pays the thermal tab the
# first one ran up.
#
# Output: wall-clock per run, md5s of the first CSV from each branch
# (to verify bit-identical cross-branch output), and the Highway target
# auto-detected by the runtime.
#
# Prerequisites: git, cmake >= 3.14, a C++17 compiler, ffmpeg, curl.
# Network access is required on first run (fetches the Tears of Steel
# mezzanine from blender.org, ~557 MB).

set -e

# --- Paths ---
WORKDIR="${WORKDIR:-$PWD}"
TOS_MOV="${TOS_MOV:-/tmp/tos.mov}"
TOS_YUV="${TOS_YUV:-/tmp/tos.yuv}"

cd "$WORKDIR"

# --- Fetch / prepare input video ---
if [ ! -f "$TOS_MOV" ]; then
    echo "Downloading Tears of Steel 1080p mezzanine..."
    curl -sSL -o "$TOS_MOV" https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov
fi

# 180s clip starting at minute 1 = minutes 1-4 of ToS. Skips the opening
# title card/static scenes which aren't representative workload.
YUV_EXPECTED_SIZE=9953280000   # 180 * 24 * 1920 * 800 * 1.5
YUV_SIZE=0
if [ -f "$TOS_YUV" ]; then
    YUV_SIZE=$(stat -c%s "$TOS_YUV" 2>/dev/null || stat -f%z "$TOS_YUV")
fi
if [ "$YUV_SIZE" != "$YUV_EXPECTED_SIZE" ]; then
    echo "Extracting 180s YUV clip (minutes 1-4)..."
    rm -f "$TOS_YUV"
    ffmpeg -hide_banner -loglevel error -ss 60 -i "$TOS_MOV" -t 180 \
        -pix_fmt yuv420p -f rawvideo "$TOS_YUV"
fi

# --- Build stable ---
echo "Building stable..."
git checkout stable
rm -rf build-stable
cmake -S . -B build-stable -DCMAKE_BUILD_TYPE=Release -DENABLE_TEST=OFF > /dev/null
cmake --build build-stable -j > /dev/null

# --- Build highway ---
echo "Building highway..."
git checkout highway
rm -rf build-highway
cmake -S . -B build-highway -DCMAKE_BUILD_TYPE=Release -DENABLE_TEST=OFF > /dev/null
cmake --build build-highway -j > /dev/null

# --- Benchmark ---
ARGS="--input $TOS_YUV --input-res 1920x800 --input-depth 8 --input-fps 24 --input-csp 420"

# Seconds to sleep between runs so the machine can return to a
# comparable thermal baseline. 30 s is enough for Apple silicon and
# mainstream desktop x86 to shed most accumulated heat from a short
# (< 20 s) run. Override with COOLDOWN=0 for a back-to-back run.
COOLDOWN="${COOLDOWN:-30}"

run() {
    local branch="$1" idx="$2"
    local bin="./build-$branch/source/apps/vca/vca"
    local csv="/tmp/bench-$branch-$idx.csv"
    local t
    t=$({ /usr/bin/time -p "$bin" $ARGS --complexity-csv "$csv" > /dev/null; } 2>&1 \
        | awk '/^real/ {print $2}')
    echo "  $branch run $idx: real=${t}s"
}

cooldown() {
    if [ "$COOLDOWN" -gt 0 ]; then
        echo "  (cooldown ${COOLDOWN}s)"
        sleep "$COOLDOWN"
    fi
}

echo
echo "=== Warmup (discarded) ==="
run stable 0
cooldown
run highway 0
cooldown

echo
echo "=== Interleaved runs (stable, highway, stable, highway, ...) ==="
for i in 1 2 3 4 5; do
    run stable $i
    cooldown
    run highway $i
    [ "$i" -lt 5 ] && cooldown
done

echo
echo "=== CSV identity check ==="
if command -v md5sum > /dev/null; then
    md5sum /tmp/bench-stable-1.csv /tmp/bench-highway-1.csv
else
    md5 /tmp/bench-stable-1.csv /tmp/bench-highway-1.csv
fi

echo
echo "=== Highway SIMD target auto-detected ==="
./build-highway/source/apps/vca/vca $ARGS --complexity-csv /tmp/bench-target.csv 2>&1 \
    | grep -E "Using SIMD target" || true
rm -f /tmp/bench-target.csv
