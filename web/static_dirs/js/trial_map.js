/**
 * Clinical Trial Map — Dynamic Kenney Tiny Town tileset
 *
 * Loads map_meta.json for dynamic home/hospital positions and waypoint graph.
 * Uses BFS pathfinding along waypoints for character movement.
 */

const TILE = 16;
const STATIC = '/static/assets';

// Character sprite indices grouped by gender + skin tone (PIPOYA civilian sprites)
const SPRITE_MAP = {
  F: {
    light:  [1,5,9,15,25,29,49,62,66,69,70,73,74,77,78,79,81,82,85,86,87,89,92,93,95],
    medium: [2,12,14,53,61,63,65,67,71,75,90],
    dark:   [3,31,51,54,64,68,72,76,80,83,84,88,91,94,96,97],
  },
  M: {
    light:  [6,7,8,10,11,17,21,24,32,33,36,37,40,41,45,52,55,58],
    medium: [13,16,18,20,26,28,30,34,38,42,44,46,48,50,56,59],
    dark:   [4,19,22,23,27,35,39,43,47,57,60],
  },
};
// Flat list of ALL indices (sorted for atlas mapping)
const SPRITE_INDICES = [].concat(...Object.values(SPRITE_MAP.F), ...Object.values(SPRITE_MAP.M));
// Sorted unique indices — position in this array = cell position in atlas
const _SORTED_INDICES = [...new Set(SPRITE_INDICES)].sort((a, b) => a - b);
const _ATLAS_COLS = 10;
const _ATLAS_CELL_W = 96;   // 3 cols × 32px per character sheet
const _ATLAS_CELL_H = 128;  // 4 rows × 32px per character sheet
// Map sprite index → atlas cell position
const _ATLAS_POS = {};
_SORTED_INDICES.forEach((idx, pos) => { _ATLAS_POS[idx] = pos; });

// Per-run sprite offset — set via setSpriteOffset(runId) from each page
let _spriteOffset = 0;
function setSpriteOffset(runId) {
  let h = 0;
  for (let i = 0; i < runId.length; i++) h = ((h << 5) - h + runId.charCodeAt(i)) | 0;
  _spriteOffset = Math.abs(h);
}

/** Pick sprite index matching patient sex + race */
function _pickSprite(p, fallbackIdx) {
  const sex = ((p.sex || '').charAt(0) || 'M').toUpperCase();
  const gender = sex === 'F' ? 'F' : 'M';
  const race = (p.race || '').toUpperCase();
  const tone = race.includes('BLACK') ? 'dark'
             : race.includes('ASIAN') ? 'medium'
             : race.includes('WHITE') ? 'light'
             : 'medium';
  const pool = (SPRITE_MAP[gender] && SPRITE_MAP[gender][tone]) || SPRITE_MAP.M.light;
  const num = parseInt((p.patient_id || '').replace(/\D/g, ''), 10) || (fallbackIdx + 1);
  return pool[(_spriteOffset + num - 1) % pool.length];
}

const STATUS_COLORS = {
  severe:   0xf85149,
  moderate: 0xd29922,
  normal:   0x3fb950,
};

/** Build speech bubble text from actual patient AE data */
function _buildBubbleText(p) {
  const aes = p.active_aes || [];
  if (aes.length === 0) return null;
  // Pick the highest-grade AE
  const worst = aes.reduce((a, b) => (b.grade || 0) > (a.grade || 0) ? b : a, aes[0]);
  const term = (worst.term || '?').replace(/_/g, ' ');
  return `${term} G${worst.grade}`;
}

class TrialMap {
  constructor(containerId, patients, onPatientClick, runId, options) {
    this.patients = patients || [];
    this.onPatientClick = onPatientClick;
    this.runId = runId;
    this.options = options || {};
    this.sprites = {};
    this.nameLabels = {};
    this.statusDots = {};
    this.speechBubbles = {};
    this.bubbleTimers = {};
    this.tombstones = {};
    this._scene = null;
    this._containerId = containerId;
    this.onPhoneCall = null;
    this._animating = false;  // true while movement tweens are active
    this._animDoneResolve = null;

    // Dynamic map data (loaded from API)
    this.mapMeta = null;
    this.homePositions = {};
    this.hospitalPos = null;
    this.waypointGraph = {};

    // Wait for both map meta and web font to load before starting Phaser
    Promise.all([
      this._loadMapMeta(),
      document.fonts.load('16px "NeoDunggeunmo"'),
    ]).then(() => {
      this.game = new Phaser.Game({
        type: Phaser.AUTO,
        parent: containerId,
        backgroundColor: '#63c74d',
        pixelArt: true,
        scale: {
          mode: Phaser.Scale.RESIZE,
          autoCenter: Phaser.Scale.CENTER_BOTH,
        },
        scene: {
          preload: this._preload.bind(this),
          create:  this._create.bind(this),
          update:  this._update.bind(this),
        },
      });
    });
  }

  async _loadMapMeta() {
    try {
      const resp = await fetch(`/api/map/${this.runId}/`);
      if (resp.ok) {
        this.mapMeta = await resp.json();
        this.hospitalPos = this.mapMeta.hospital;

        const homes = this.mapMeta.homes || [];
        homes.forEach(h => {
          this.homePositions[h.patient_idx] = h;
        });

        (this.mapMeta.waypoints || []).forEach(wp => {
          this.waypointGraph[wp.id] = {
            x: wp.x, y: wp.y,
            neighbors: wp.neighbors || [],
          };
        });
      }
    } catch (e) {
      console.warn('Failed to load map metadata, using defaults:', e);
    }

    if (!this.hospitalPos) {
      this.hospitalPos = { x: 4, y: 14, waypoint_id: 'hosp' };
    }
  }

  _getHomePos(patientIdx) {
    if (this.homePositions[patientIdx]) {
      const h = this.homePositions[patientIdx];
      return { x: h.x, y: h.y };
    }
    const col = patientIdx % 5;
    const row = Math.floor(patientIdx / 5);
    return { x: 17 + col * 5, y: 6 + row * 20 };
  }

  // ─── BFS Pathfinding ───────────────────────────────────
  findPath(fromWpId, toWpId) {
    if (!fromWpId || !toWpId || fromWpId === toWpId) return [];
    if (!this.waypointGraph[fromWpId] || !this.waypointGraph[toWpId]) return [];

    const queue = [[fromWpId, [fromWpId]]];
    const visited = new Set([fromWpId]);

    while (queue.length > 0) {
      const [current, path] = queue.shift();
      if (current === toWpId) {
        return path.map(id => ({
          id,
          x: this.waypointGraph[id].x * TILE + TILE / 2,
          y: this.waypointGraph[id].y * TILE + TILE / 2,
        }));
      }
      for (const nb of (this.waypointGraph[current]?.neighbors || [])) {
        if (!visited.has(nb)) {
          visited.add(nb);
          queue.push([nb, [...path, nb]]);
        }
      }
    }
    return [];
  }

  /** Max visible patients inside hospital */
  static HOSP_MAX_VISIBLE = 6;

  /**
   * Compute position inside hospital building for the n-th visitor.
   * 3 cols × 2 rows grid inside the building. Returns null if over limit.
   */
  _getHospSlot(visitIdx) {
    if (visitIdx >= TrialMap.HOSP_MAX_VISIBLE) return null; // hide overflow
    const cols = 3;
    const spacingX = 12;
    const spacingY = 11;
    const col = visitIdx % cols;
    const row = Math.floor(visitIdx / cols);
    const baseX = this.hospitalPos.x * TILE + TILE / 2;
    const baseY = this.hospitalPos.y * TILE - 14;
    const gridW = (cols - 1) * spacingX;
    const x = baseX - gridW / 2 + col * spacingX;
    const y = baseY - row * spacingY;
    return { x, y };
  }

  /**
   * Update the hospital patient count badge.
   */
  _updateHospBadge(totalInHosp) {
    const scene = this._scene;
    if (!scene) return;
    if (this._hospBadge) { this._hospBadge.destroy(); this._hospBadge = null; }
    if (this._hospBadgeBg) { this._hospBadgeBg.destroy(); this._hospBadgeBg = null; }
    if (totalInHosp <= 0) return;

    const bx = this.hospitalPos.x * TILE + TILE / 2 + 30;
    const by = this.hospitalPos.y * TILE - 40;
    const label = totalInHosp > TrialMap.HOSP_MAX_VISIBLE
      ? `${totalInHosp}` : `${totalInHosp}`;

    this._hospBadgeBg = scene.add.circle(bx, by, 8, 0x0d1117);
    this._hospBadgeBg.setDepth(25).setAlpha(0.85);
    this._hospBadge = scene.add.text(bx, by, label, {
      fontFamily: '"NeoDunggeunmo", monospace', fontSize: '9px', fontStyle: 'bold',
      color: '#f0f6fc', stroke: '#0d1117', strokeThickness: 1,
      resolution: 4,
    });
    this._hospBadge.setOrigin(0.5, 0.5).setDepth(26);
  }

  // ─── Preload ─────────────────────────────────────────
  _preload() {
    const s = this.game.scene.scenes[0];
    // Tilemap from per-run API (sized to actual patient count)
    s.load.tilemapTiledJSON('map', `/api/map/${this.runId}/tilemap/?v=4`);
    s.load.image('tiny-town', `${STATIC}/tiles/kenney/tiny-town/Tilemap/tilemap_packed.png`);
    // Custom building sprites
    s.load.image('hospital-building', `${STATIC}/buildings/hospital.png`);
    ['green', 'orange', 'purple', 'red'].forEach(c => s.load.image(`house-${c}`, `${STATIC}/buildings/house_${c}.png`));
    // Tombstone spritesheet (5 cols × 2 rows, 16×24 per frame)
    s.load.spritesheet('tombs', `${STATIC}/decorations/tombs.png`, { frameWidth: 16, frameHeight: 24 });
    // Flower & stone decorations (5×4 grid, 16×16 per frame)
    s.load.spritesheet('flower-stones', `${STATIC}/decorations/flower_stones.png`, { frameWidth: 16, frameHeight: 16 });
    // Emotion icons (2×4 grid, 16×16) — left col: None, G1, G2, G3+
    s.load.spritesheet('emotions', `${STATIC}/characters/emotions.png`, { frameWidth: 16, frameHeight: 16 });
    // Skull icon for deceased patients (16×16)
    s.load.image('skull', `${STATIC}/decorations/skull.png`);
    // Custom environment sprites
    s.load.image('trees-atlas', `${STATIC}/decorations/trees.png`);
    s.load.image('plants-atlas', `${STATIC}/decorations/plants.png`);
    // Auto-tile road & grass spritesheets
    s.load.spritesheet('road-autotile', `${STATIC}/tiles/road_autotile.png`, { frameWidth: 16, frameHeight: 16 });
    s.load.spritesheet('grass-tiles', `${STATIC}/tiles/grass_tiles.png`, { frameWidth: 16, frameHeight: 16 });
    // Character atlas (97 sprites in one image — 98 HTTP requests → 1)
    s.load.image('patient-atlas', `${STATIC}/characters/patient_atlas.png`);
  }

  // ─── Create ──────────────────────────────────────────
  _create() {
    this._scene = this.game.scene.scenes[0];
    const scene = this._scene;

    // ── Extract individual spritesheets from atlas ──
    const atlasTex = scene.textures.get('patient-atlas');
    const atlasSource = atlasTex.source[0];
    _SORTED_INDICES.forEach((idx, pos) => {
      const col = pos % _ATLAS_COLS;
      const row = Math.floor(pos / _ATLAS_COLS);
      const key = `patient_${String(idx).padStart(2, '0')}`;
      const sx = col * _ATLAS_CELL_W;
      const sy = row * _ATLAS_CELL_H;
      // Create a spritesheet texture from the atlas region (3×4 grid of 32×32 frames)
      const canvas = document.createElement('canvas');
      canvas.width = _ATLAS_CELL_W;
      canvas.height = _ATLAS_CELL_H;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(atlasSource.image, sx, sy, _ATLAS_CELL_W, _ATLAS_CELL_H, 0, 0, _ATLAS_CELL_W, _ATLAS_CELL_H);
      scene.textures.addSpriteSheet(key, canvas, { frameWidth: 32, frameHeight: 32 });
    });

    // Load tilemap for layout data (positions, tile types)
    const map = scene.make.tilemap({ key: 'map' });
    const tileset = map.addTilesetImage('tiny-town', 'tiny-town', 16, 16, 0, 0);
    if (!tileset) { console.error('Failed to load tileset'); return; }

    // Create layers (hidden — we draw custom visuals on top)
    const layers = {};
    ['Ground', 'Buildings', 'Roofs', 'Decor'].forEach((name, i) => {
      try {
        const layer = map.createLayer(name, tileset, 0, 0);
        if (layer) { layer.setDepth(i); layer.setVisible(false); layers[name] = layer; }
      } catch (e) { console.warn(`Layer "${name}":`, e.message); }
    });

    // ── Custom environment rendering ──
    const ROAD_IDS = new Set([13,14,15,25,26,27,37,38,39]);
    // Build occupied tile lookup (roads + buildings) for decoration collision check
    this._occupiedTiles = new Set();

    // Grass background — tile-based with variants
    let rng = 42;
    const nextRng = () => { rng = (rng * 1103515245 + 12345) & 0x7fffffff; return rng; };
    for (let gy = 0; gy < map.height; gy++) {
      for (let gx = 0; gx < map.width; gx++) {
        const frame = nextRng() % 16;
        const px = gx * TILE + TILE / 2;
        const py = gy * TILE + TILE / 2;
        scene.add.image(px, py, 'grass-tiles', frame).setDepth(-2);
      }
    }

    // Roads from Ground layer data — 9-patch auto-tile with bitmask
    // Bitmask: bit0=UP(1), bit1=RIGHT(2), bit2=DOWN(4), bit3=LEFT(8)
    // Tileset layout (4x4 = 16 frames):
    //   TL(0)  T-a(1)  T-b(2)  TR(3)
    //   L-a(4) C-a(5)  C-b(6)  R-a(7)
    //   L-b(8) C-c(9)  C-d(10) R-b(11)
    //   BL(12) B-a(13) B-b(14) BR(15)
    const roadFrame = (mask, x, y) => {
      const px = x & 1, py = y & 1;
      switch (mask) {
        case  6: return 0;                          // TL corner (RIGHT+DOWN)
        case 12: return 3;                          // TR corner (DOWN+LEFT)
        case  3: return 12;                         // BL corner (UP+RIGHT)
        case  9: return 15;                         // BR corner (UP+LEFT)
        case 14: return px ? 2 : 1;                 // top edge (R+D+L)
        case 11: return px ? 14 : 13;               // bottom edge (U+R+L)
        case  7: return py ? 8 : 4;                 // left edge (U+R+D)
        case 13: return py ? 11 : 7;                // right edge (U+D+L)
        case 15: return [5, 6, 9, 10][py * 2 + px]; // center (all)
        case  5: return py ? 9 : 5;                 // vertical straight (U+D)
        case 10: return px ? 6 : 5;                 // horizontal straight (R+L)
        case  4: return px ? 2 : 1;                 // dead-end down → top cap
        case  1: return px ? 14 : 13;               // dead-end up → bottom cap
        case  2: return py ? 8 : 4;                 // dead-end right → left cap
        case  8: return py ? 11 : 7;                // dead-end left → right cap
        default: return 5;                          // isolated → center fill
      }
    };
    const groundLayer = layers['Ground'];
    if (groundLayer) {
      const gData = groundLayer.layer.data;
      const isRoad = (x, y) => {
        if (y < 0 || y >= map.height || x < 0 || x >= map.width) return false;
        const t = gData[y][x];
        return t && ROAD_IDS.has(t.index);
      };
      // Pre-compute inner corner positions (grass tiles at intersection notches)
      const innerCorners = new Set();
      for (let y = 0; y < map.height; y++) {
        for (let x = 0; x < map.width; x++) {
          if (isRoad(x, y)) continue;
          const rd = isRoad(x + 1, y), ld = isRoad(x - 1, y);
          const dd = isRoad(x, y + 1), ud = isRoad(x, y - 1);
          if ((dd && rd && isRoad(x + 1, y + 1)) ||
              (dd && ld && isRoad(x - 1, y + 1)) ||
              (ud && rd && isRoad(x + 1, y - 1)) ||
              (ud && ld && isRoad(x - 1, y - 1))) {
            innerCorners.add(`${x},${y}`);
          }
        }
      }
      // Treat inner corners as road so adjacent tiles use center frames (5,6,9,10)
      const isRoadEx = (x, y) => isRoad(x, y) || innerCorners.has(`${x},${y}`);
      for (let y = 0; y < map.height; y++) {
        for (let x = 0; x < map.width; x++) {
          const tile = gData[y][x];
          if (tile && ROAD_IDS.has(tile.index)) {
            this._occupiedTiles.add(`${x},${y}`);
            const rx = x * TILE + TILE / 2;
            const ry = y * TILE + TILE / 2;
            let mask = 0;
            if (isRoadEx(x, y - 1)) mask |= 1;  // up
            if (isRoadEx(x + 1, y)) mask |= 2;  // right
            if (isRoadEx(x, y + 1)) mask |= 4;  // down
            if (isRoadEx(x - 1, y)) mask |= 8;  // left
            scene.add.image(rx, ry, 'road-autotile', roadFrame(mask, x, y)).setDepth(0);
          }
        }
      }
      // Inner corner overlays — use outer corner frames (0,3,12,15) on grass
      for (let y = 0; y < map.height; y++) {
        for (let x = 0; x < map.width; x++) {
          if (!innerCorners.has(`${x},${y}`)) continue;
          const rx = x * TILE + TILE / 2;
          const ry = y * TILE + TILE / 2;
          const rd = isRoad(x + 1, y), ld = isRoad(x - 1, y);
          const dd = isRoad(x, y + 1), ud = isRoad(x, y - 1);
          if (dd && rd) scene.add.image(rx, ry, 'road-autotile', 0).setDepth(0);
          if (dd && ld) scene.add.image(rx, ry, 'road-autotile', 3).setDepth(0);
          if (ud && rd) scene.add.image(rx, ry, 'road-autotile', 12).setDepth(0);
          if (ud && ld) scene.add.image(rx, ry, 'road-autotile', 15).setDepth(0);
        }
      }
    }

    // Mark building & roof tiles as occupied
    ['Buildings', 'Roofs'].forEach(layerName => {
      const layer = layers[layerName];
      if (!layer) return;
      const d = layer.layer.data;
      for (let y = 0; y < map.height; y++)
        for (let x = 0; x < map.width; x++)
          if (d[y][x] && d[y][x].index > 0) this._occupiedTiles.add(`${x},${y}`);
    });

    // Tree & plant sprite frames from atlas
    const treeTex = scene.textures.get('trees-atlas');
    treeTex.add('tree_0', 0, 48, 28, 16, 36);
    treeTex.add('tree_1', 0, 65, 28, 14, 36);
    treeTex.add('tree_2', 0, 95, 28, 18, 36);

    const plantTex = scene.textures.get('plants-atlas');
    const pCols = [[1,15],[17,31],[33,47],[49,63],[65,78],[80,95],[98,111],[114,127]];
    pCols.forEach((c, i) => plantTex.add(`p_${i}`, 0, c[0], 18, c[1] - c[0] + 1, 13));
    pCols.forEach((c, i) => plantTex.add(`p_${8 + i}`, 0, c[0], 34, c[1] - c[0] + 1, 13));

    // Building proximity check — used for trees, plants, and flower decorations
    const _buildingCenters = [];
    this.patients.forEach((p, i) => {
      const h = this._getHomePos(i);
      _buildingCenters.push({ x: h.x * TILE + TILE / 2, y: h.y * TILE - 28 });
    });
    _buildingCenters.push({ x: this.hospitalPos.x * TILE + TILE / 2, y: this.hospitalPos.y * TILE - 30 });
    const BLDG_RADIUS = 50;
    const nearBuilding = (px, py) => {
      for (const b of _buildingCenters) {
        const dx = px - b.x, dy = py - b.y;
        if (dx * dx + dy * dy < BLDG_RADIUS * BLDG_RADIUS) return true;
      }
      return false;
    };

    // Shared placement tracker — prevents overlap between trees, plants, flowers
    const _placed = [];
    const SPACING = 14;  // minimum pixel distance between any two decorations
    const canPlace = (px, py) => {
      for (const p of _placed) {
        const dx = px - p.x, dy = py - p.y;
        if (dx * dx + dy * dy < SPACING * SPACING) return false;
      }
      return true;
    };
    const markPlaced = (px, py) => _placed.push({ x: px, y: py });

    // Decor: place trees & plants from tilemap data (skip near buildings & overlaps)
    const TREE_TILE_IDS = new Set([5, 6, 7]);
    const decorLayer = layers['Decor'];
    if (decorLayer) {
      const dData = decorLayer.layer.data;
      for (let y = 0; y < map.height; y++) {
        for (let x = 0; x < map.width; x++) {
          const tile = dData[y][x];
          if (tile && tile.index > 0) {
            const px = x * TILE + TILE / 2;
            const py = y * TILE + TILE;
            if (nearBuilding(px, py)) continue;
            if (!canPlace(px, py)) continue;
            const r = nextRng();
            if (TREE_TILE_IDS.has(tile.index)) {
              const tree = scene.add.image(px, py, 'trees-atlas', `tree_${r % 3}`);
              tree.setScale(0.9 + (r % 4) * 0.1);
              tree.setOrigin(0.5, 1);
              tree.setDepth(3);
            } else {
              const plant = scene.add.image(px, py, 'plants-atlas', `p_${r % 16}`);
              plant.setOrigin(0.5, 1);
              plant.setDepth(2);
            }
            markPlaced(px, py);
          }
        }
      }
    }

    // Camera — zoom to content bounds (actual buildings) instead of full tilemap
    const camera = scene.cameras.main;
    camera.setBounds(0, 0, map.widthInPixels, map.heightInPixels);
    const cb = this.mapMeta && this.mapMeta.content_bounds;
    if (cb) {
      const bw = (cb.max_x - cb.min_x + 1) * TILE;
      const bh = (cb.max_y - cb.min_y + 1) * TILE;
      const fitZoom = Math.min(this.game.config.width / bw, this.game.config.height / bh) * 0.9;
      camera.setZoom(Math.max(fitZoom, 1.0));
      // Center on hospital when requested (landing page), otherwise on content bounds center
      if (this.options.centerOnHospital && this.hospitalPos) {
        camera.centerOn(this.hospitalPos.x * TILE + TILE / 2, this.hospitalPos.y * TILE + TILE / 2);
      } else {
        camera.centerOn((cb.min_x + cb.max_x + 1) / 2 * TILE, (cb.min_y + cb.max_y + 1) / 2 * TILE);
      }
    } else {
      const fitZoom = Math.min(this.game.config.width / map.widthInPixels, this.game.config.height / map.heightInPixels) * 0.95;
      camera.setZoom(Math.max(fitZoom, 1.0));
      camera.centerOn(this.hospitalPos.x * TILE + TILE / 2, this.hospitalPos.y * TILE + TILE / 2);
    }

    // Drag to pan
    scene.input.on('pointermove', (pointer) => {
      if (pointer.isDown) {
        camera.scrollX -= (pointer.x - pointer.prevPosition.x) / camera.zoom;
        camera.scrollY -= (pointer.y - pointer.prevPosition.y) / camera.zoom;
      }
    });
    scene.input.on('wheel', (p, go, dx, dy) => {
      camera.setZoom(Phaser.Math.Clamp(camera.zoom - dy * 0.003, 0.5, 5.0));
    });

    this._createAnimations(scene);

    // ── Place patients ──
    let hospVisitIdx = 0;
    const totalInHosp = this.patients.filter(p => {
      const l = (p.location || '').toUpperCase();
      return l === 'HOSPITAL' || l === 'OUTPATIENT' || l === 'INPATIENT';
    }).length;

    this.patients.forEach((p, i) => {
      const spriteIdx = _pickSprite(p, i);
      const spriteKey = `patient_${String(spriteIdx).padStart(2, '0')}`;
      const home = this._getHomePos(i);

      const loc = (p.location || '').toUpperCase();
      const isDead = loc === 'DECEASED';
      const isHosp = loc === 'HOSPITAL' || loc === 'OUTPATIENT' || loc === 'INPATIENT';
      let px, py, hidden = false;
      if (isDead) {
        px = home.x * TILE + TILE / 2;
        py = home.y * TILE + TILE + 4;  // in front of house, slightly lower
      } else if (isHosp) {
        const slot = this._getHospSlot(hospVisitIdx++);
        if (slot) {
          px = slot.x;
          py = slot.y;
        } else {
          px = this.hospitalPos.x * TILE + TILE / 2;
          py = this.hospitalPos.y * TILE;
          hidden = true;
        }
      } else {
        px = home.x * TILE + TILE / 2;
        py = home.y * TILE - 16;  // inside house (image extends upward)
      }

      const sprite = scene.add.sprite(px, py, spriteKey, 1);
      sprite.setScale(0.5);
      sprite.setDepth(10);
      if (isDead || hidden) { sprite.setVisible(false); }
      else { sprite.setAlpha(0.7); }  // translucent inside building (hospital or home)
      sprite.setInteractive({ useHandCursor: true });
      sprite.on('pointerdown', () => {
        if (this.onPatientClick) this.onPatientClick(p.patient_id);
        this.highlightPatient(p.patient_id);
      });

      const labelY = isDead ? py - 28 : py - 12;
      const label = scene.add.text(px, labelY, p.patient_id, {
        fontFamily: '"NeoDunggeunmo", monospace', fontSize: '8px', fontStyle: 'bold',
        color: isDead ? '#888888' : '#ffffff',
        stroke: '#000000', strokeThickness: 2,
        align: 'center', resolution: 4,
      });
      label.setOrigin(0.5, 1).setDepth(20);
      if (hidden) label.setVisible(false);

      let dot;
      if (isDead) {
        dot = scene.add.image(px - 18, labelY, 'skull');
        dot.setOrigin(0.5, 1).setScale(0.7).setDepth(21);
      } else {
        dot = scene.add.sprite(px - 18, labelY, 'emotions', this._getEmotionFrame(p));
        dot.setOrigin(0.5, 1).setScale(0.7).setDepth(21);
      }
      if (hidden) dot.setVisible(false);

      // Tombstone for deceased patients
      if (isDead) {
        const tombFrame = 0;
        const tomb = scene.add.sprite(px, py, 'tombs', tombFrame);
        tomb.setOrigin(0.5, 1).setDepth(6);
        this.tombstones[p.patient_id] = tomb;
      }

      const homeWpId = this.homePositions[i]?.waypoint_id || null;
      sprite.setData('homeX', home.x * TILE + TILE / 2);
      sprite.setData('homeY', home.y * TILE - 16);
      sprite.setData('homeWpId', homeWpId);
      sprite.setData('spriteKey', spriteKey);
      sprite.setData('patientIdx', i);
      sprite.setData('currentWpId', isHosp ? 'hosp' : homeWpId);

      this.sprites[p.patient_id] = sprite;
      this.nameLabels[p.patient_id] = label;
      this.statusDots[p.patient_id] = dot;
    });

    this._updateHospBadge(totalInHosp);

    // ── Hospital building sprite ──
    const hospX = this.hospitalPos.x * TILE + TILE / 2;
    const hospY = this.hospitalPos.y * TILE;
    const hospSprite = scene.add.image(hospX, hospY, 'hospital-building');
    hospSprite.setScale(0.6);
    hospSprite.setOrigin(0.5, 1);
    hospSprite.setDepth(5);

    // ── House sprites at each patient's home ──
    const houseKeys = ['house-green', 'house-orange', 'house-purple', 'house-red'];
    this.patients.forEach((p, i) => {
      const home = this._getHomePos(i);
      const hx = home.x * TILE + TILE / 2;
      const hy = home.y * TILE;
      const houseKey = houseKeys[i % houseKeys.length];
      const house = scene.add.image(hx, hy, houseKey);
      house.setScale(0.5);
      house.setOrigin(0.5, 1);
      house.setDepth(4);
    });

    // ── Flower & stone decorations — scatter across all grass ──
    const DECO_FRAMES = 20;
    let dr = 137;
    const nr = () => { dr = (dr * 1103515245 + 12345) & 0x7fffffff; return dr; };
    const isOccupied = (px, py) => {
      const tx = Math.floor(px / TILE);
      const ty = Math.floor(py / TILE);
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++)
          if (this._occupiedTiles.has(`${tx+dx},${ty+dy}`)) return true;
      if (nearBuilding(px, py)) return true;
      return false;
    };

    const decoCount = Math.floor(map.width * map.height / 12);
    for (let d = 0; d < decoCount; d++) {
      const px = (nr() % (map.widthInPixels - 32)) + 16;
      const py = (nr() % (map.heightInPixels - 32)) + 16;
      if (isOccupied(px, py)) continue;
      if (!canPlace(px, py)) continue;
      const frame = nr() % DECO_FRAMES;
      scene.add.sprite(px, py, 'flower-stones', frame)
        .setOrigin(0.5, 0.5).setDepth(2);
      markPlaced(px, py);
    }

    this.dayText = null;

    // Notify caller that scene is ready (used by landing page zoom-in)
    // Delay by 1 frame so Phaser RESIZE scale mode has settled
    if (this.options.onReady) {
      const self = this;
      scene.time.delayedCall(50, () => { self.options.onReady(self); });
    }
  }

  _createAnimations(scene) {
    SPRITE_INDICES.forEach(i => {
      const key = `patient_${String(i).padStart(2, '0')}`;
      const dirs = [
        ['down',  [0, 1, 2, 1]], ['left',  [3, 4, 5, 4]],
        ['right', [6, 7, 8, 7]], ['up',    [9, 10, 11, 10]],
      ];
      dirs.forEach(([dir, frames]) => {
        scene.anims.create({
          key: `${key}_${dir}`,
          frames: scene.anims.generateFrameNumbers(key, { frames }),
          frameRate: 6, repeat: -1,
        });
      });
    });
  }

  _update() {
    this.updateBubblePositions();
  }

  _getStatusColor(p) {
    const aes = p.active_aes || [];
    if (aes.some(ae => ae.grade >= 3)) return STATUS_COLORS.severe;
    if (aes.length > 0) return STATUS_COLORS.moderate;
    return STATUS_COLORS.normal;
  }

  /** Map max AE grade to emotion spritesheet frame (left column: 0,2,4,6) */
  _getEmotionFrame(p) {
    const aes = p.active_aes || [];
    if (aes.length === 0) return 0;            // None — frame 0
    const maxGrade = Math.max(...aes.map(ae => ae.grade || 0));
    if (maxGrade >= 3) return 6;               // G3+ — frame 6
    if (maxGrade >= 2) return 4;               // G2  — frame 4
    return 2;                                  // G1  — frame 2
  }

  // ═══ Public API ═══════════════════════════════════════

  /**
   * Returns a Promise that resolves when all movement animations are done.
   * The auto-play controller should await this before advancing.
   */
  updateDay(dayNum, patients) {
    if (!this._scene) return Promise.resolve();
    this.currentDay = dayNum;
    if (this.dayText) this.dayText.setText(`DAY ${dayNum}`);

    let maxAnimDuration = 0;

    // Assign hospital slot indices to patients at hospital
    let hospVisitIdx = 0;
    const hospSlots = {};
    patients.forEach(p => {
      const loc = (p.location || '').toUpperCase();
      if (loc === 'HOSPITAL' || loc === 'OUTPATIENT' || loc === 'INPATIENT') {
        hospSlots[p.patient_id] = hospVisitIdx++;
      }
    });
    const totalInHosp = hospVisitIdx;
    this._updateHospBadge(totalInHosp);

    patients.forEach((p, i) => {
      const sprite = this.sprites[p.patient_id];
      if (!sprite) return;

      const idx = sprite.getData('patientIdx');
      const spriteKey = sprite.getData('spriteKey');
      const home = this._getHomePos(idx);
      const homeWpId = sprite.getData('homeWpId');
      const currentWpId = sprite.getData('currentWpId');

      const loc = (p.location || '').toUpperCase();
      const isDead = loc === 'DECEASED';
      const isHosp = loc === 'HOSPITAL' || loc === 'OUTPATIENT' || loc === 'INPATIENT';
      const targetWpId = isHosp ? 'hosp' : homeWpId;

      // Handle newly deceased patients
      if (isDead) {
        if (!this.tombstones[p.patient_id]) {
          // Immediately hide character sprite (no walk home)
          this._scene.tweens.killTweensOf(sprite);
          sprite.setAlpha(0);
          sprite.setVisible(false);
          // Hide emotion icon immediately
          const oldDot = this.statusDots[p.patient_id];
          if (oldDot) oldDot.destroy();
          this.statusDots[p.patient_id] = null;
          // Place tombstone in front of house with fade-in
          const tx = home.x * TILE + TILE / 2;
          const ty = home.y * TILE + TILE + 4;
          const tombFrame = 0;
          const tomb = this._scene.add.sprite(tx, ty, 'tombs', tombFrame);
          tomb.setOrigin(0.5, 1).setDepth(6);
          tomb.setAlpha(0);
          this._scene.tweens.add({ targets: tomb, alpha: 1, duration: 600 });
          this.tombstones[p.patient_id] = tomb;
          // Move label above tombstone and grey out
          const label = this.nameLabels[p.patient_id];
          if (label) {
            label.setX(tx);
            label.setY(ty - 28);
            label.setColor('#888888');
          }
          // Skull icon at tombstone
          const skull = this._scene.add.image(tx - 18, ty - 28, 'skull');
          skull.setOrigin(0.5, 1).setScale(0.7).setDepth(21).setAlpha(0);
          this._scene.tweens.add({ targets: skull, alpha: 1, duration: 600 });
          this.statusDots[p.patient_id] = skull;
        }
        return; // skip movement for dead patients
      }

      let tx, ty;
      const hidden = isHosp && hospSlots[p.patient_id] >= TrialMap.HOSP_MAX_VISIBLE;
      if (isHosp) {
        const slot = this._getHospSlot(hospSlots[p.patient_id]);
        if (slot) {
          tx = slot.x;
          ty = slot.y;
        } else {
          tx = this.hospitalPos.x * TILE + TILE / 2;
          ty = this.hospitalPos.y * TILE;
        }
      } else {
        tx = home.x * TILE + TILE / 2;
        ty = home.y * TILE - 16;  // inside house
      }

      const label = this.nameLabels[p.patient_id];
      const dot = this.statusDots[p.patient_id];

      const dx = tx - sprite.x;
      const dy = ty - sprite.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist > 2) {
        // Keep visible during walk, even if overflow — hide after arrival
        sprite.setVisible(true);
        sprite.setAlpha(0.7);
        if (label) label.setVisible(true);
        if (dot) dot.setVisible(true);

        const path = (currentWpId && targetWpId)
          ? this.findPath(currentWpId, targetWpId) : [];

        let animDur = 0;
        if (path.length >= 2) {
          animDur = this._walkAlongPath(sprite, p.patient_id, path, tx, ty, spriteKey, true);
        } else {
          animDur = this._directTween(sprite, p.patient_id, tx, ty, spriteKey, dist, true);
        }

        // Hide overflow patients after animation completes
        if (hidden) {
          this._scene.time.delayedCall(animDur, () => {
            sprite.setVisible(false);
            sprite.setAlpha(0);
            if (label) label.setVisible(false);
            if (dot) dot.setVisible(false);
          });
        }

        if (animDur > maxAnimDuration) maxAnimDuration = animDur;
        sprite.setData('currentWpId', targetWpId);
      } else {
        // Already at destination — apply visibility immediately
        sprite.setVisible(!hidden);
        sprite.setAlpha(hidden ? 0 : 0.7);
        if (label) label.setVisible(!hidden);
        if (dot) dot.setVisible(!hidden);
      }

      if (dot) { dot.setFrame(this._getEmotionFrame(p)); dot.clearTint(); }
    });

    // Return a promise that resolves when all animations finish
    this._animating = maxAnimDuration > 0;
    if (this._animating) {
      return new Promise(resolve => {
        this._scene.time.delayedCall(maxAnimDuration + 100, () => {
          this._animating = false;
          resolve();
        });
      });
    }
    return Promise.resolve();
  }

  /** Returns total animation duration in ms */
  _walkAlongPath(sprite, pid, path, finalX, finalY, spriteKey, faceDown = false) {
    const points = path.slice(1).map(wp => ({ x: wp.x, y: wp.y }));
    if (points.length > 0) {
      points[points.length - 1] = { x: finalX, y: finalY };
    }

    let totalDelay = 0;
    let prevX = sprite.x, prevY = sprite.y;
    const SPEED = 4;  // pixels per ms (faster for auto mode)

    points.forEach((pt, i) => {
      const segDx = pt.x - prevX;
      const segDy = pt.y - prevY;
      const segDist = Math.sqrt(segDx * segDx + segDy * segDy);
      const dur = Math.max(Math.min(segDist * SPEED, 800), 80);

      let dir;
      if (Math.abs(segDx) > Math.abs(segDy)) {
        dir = segDx > 0 ? 'right' : 'left';
      } else {
        dir = segDy > 0 ? 'down' : 'up';
      }

      const isLast = i === points.length - 1;

      this._scene.time.delayedCall(totalDelay, () => {
        sprite.play(`${spriteKey}_${dir}`, true);
        this._scene.tweens.add({
          targets: sprite, x: pt.x, y: pt.y,
          duration: dur, ease: 'Linear',
          onComplete: () => {
            if (isLast) {
              sprite.stop();
              const idleFrame = { down: 1, left: 4, right: 7, up: 10 };
              sprite.setFrame(faceDown ? 1 : idleFrame[dir]);
            }
          },
        });

        const label = this.nameLabels[pid];
        const dot = this.statusDots[pid];
        if (label) this._scene.tweens.add({ targets: label, x: pt.x, y: pt.y - 12, duration: dur, ease: 'Linear' });
        if (dot) this._scene.tweens.add({ targets: dot, x: pt.x - 18, y: pt.y - 12, duration: dur, ease: 'Linear' });
      });

      totalDelay += dur;
      prevX = pt.x;
      prevY = pt.y;
    });

    return totalDelay;
  }

  /** Returns animation duration in ms */
  _directTween(sprite, pid, tx, ty, spriteKey, dist, faceDown = false) {
    const dx = tx - sprite.x;
    const dy = ty - sprite.y;
    let dir;
    if (Math.abs(dx) > Math.abs(dy)) {
      dir = dx > 0 ? 'right' : 'left';
    } else {
      dir = dy > 0 ? 'down' : 'up';
    }
    sprite.play(`${spriteKey}_${dir}`, true);
    const dur = Math.max(Math.min(dist * 5, 1200), 100);
    this._scene.tweens.add({
      targets: sprite, x: tx, y: ty, duration: dur, ease: 'Power2',
      onComplete: () => {
        sprite.stop();
        const idleFrame = { down: 1, left: 4, right: 7, up: 10 };
        sprite.setFrame(faceDown ? 1 : idleFrame[dir]);
      },
    });
    const label = this.nameLabels[pid];
    const dot = this.statusDots[pid];
    if (label) this._scene.tweens.add({ targets: label, x: tx, y: ty - 12, duration: dur, ease: 'Power2' });
    if (dot) this._scene.tweens.add({ targets: dot, x: tx - 18, y: ty - 12, duration: dur, ease: 'Power2' });
    return dur;
  }

  /** true if characters are still moving */
  isAnimating() { return this._animating; }

  highlightPatient(patientId) {
    if (!this._scene) return;
    Object.values(this.sprites).forEach(s =>
      this._scene.tweens.add({ targets: s, scale: 0.5, duration: 150 })
    );
    const sprite = this.sprites[patientId];
    if (sprite) {
      this._scene.tweens.add({ targets: sprite, scale: 0.8, duration: 150 });
      this._scene.cameras.main.pan(sprite.x, sprite.y, 400, 'Power2');
    }
  }

  focusHospital() {
    if (!this._scene) return;
    const cam = this._scene.cameras.main;
    cam.pan(this.hospitalPos.x * TILE, this.hospitalPos.y * TILE, 400, 'Power2');
    cam.zoomTo(3, 400);
  }

  setZoom(zoom) {
    if (!this._scene) return;
    this._scene.cameras.main.zoomTo(Phaser.Math.Clamp(zoom, 0.5, 5.0), 300);
  }

  fitMap() {
    if (!this._scene) return;
    const cam = this._scene.cameras.main;
    const cb = this.mapMeta && this.mapMeta.content_bounds;
    if (cb) {
      const bw = (cb.max_x - cb.min_x + 1) * TILE;
      const bh = (cb.max_y - cb.min_y + 1) * TILE;
      const fit = Math.min(this.game.config.width / bw, this.game.config.height / bh) * 0.9;
      cam.zoomTo(Math.max(fit, 1.0), 400);
      cam.pan((cb.min_x + cb.max_x + 1) / 2 * TILE, (cb.min_y + cb.max_y + 1) / 2 * TILE, 400, 'Power2');
    } else {
      const child = this._scene.children.list.find(c => c.tilemap);
      if (child && child.tilemap) {
        const m = child.tilemap;
        const zx = this.game.config.width / m.widthInPixels;
        const zy = this.game.config.height / m.heightInPixels;
        cam.zoomTo(Math.max(Math.min(zx, zy) * 0.95, 1.0), 400);
        cam.pan(m.widthInPixels / 2, m.heightInPixels / 2, 400, 'Power2');
      }
    }
  }

  destroy() {
    if (this.game) this.game.destroy(true);
  }

  // ═══ Speech Bubbles ═════════════════════════════════════

  showSpeechBubble(patientId, text, duration = 4000) {
    if (!this._scene) return;
    this.clearSpeechBubble(patientId);
    const sprite = this.sprites[patientId];
    if (!sprite) return;

    const displayText = text.length > 24 ? text.substring(0, 22) + '...' : text;
    const tempText = this._scene.add.text(0, 0, displayText, { fontFamily: '"NeoDunggeunmo", monospace', fontSize: '8px', color: '#fff', resolution: 4 });
    const tw = tempText.width, th = tempText.height;
    tempText.destroy();
    const padding = 4;
    const bgWidth = tw + padding * 2, bgHeight = th + padding * 2;

    // Use a container so it follows the sprite
    const container = this._scene.add.container(sprite.x, sprite.y - 20);
    container.setDepth(50);

    const bg = this._scene.add.graphics();
    bg.fillStyle(0x1b1f23, 0.92);
    bg.lineStyle(1, 0x39d2c0, 0.7);
    bg.fillRoundedRect(-bgWidth / 2, -bgHeight - 4, bgWidth, bgHeight, 3);
    bg.strokeRoundedRect(-bgWidth / 2, -bgHeight - 4, bgWidth, bgHeight, 3);
    bg.fillStyle(0x1b1f23, 0.92);
    bg.fillTriangle(-3, -4, 3, -4, 0, 0);

    const label = this._scene.add.text(0, -bgHeight / 2 - 4, displayText, {
      fontFamily: '"NeoDunggeunmo", monospace', fontSize: '8px', color: '#e6edf3',
      align: 'center', resolution: 4,
    });
    label.setOrigin(0.5, 0.5);

    container.add([bg, label]);
    container.setAlpha(0);
    this._scene.tweens.add({ targets: container, alpha: 1, duration: 200 });

    this.speechBubbles[patientId] = { container, sprite };
    this.bubbleTimers[patientId] = this._scene.time.delayedCall(duration, () => {
      this.clearSpeechBubble(patientId);
    });
  }

  /** Called every frame to keep bubbles following sprites */
  updateBubblePositions() {
    for (const [pid, bubble] of Object.entries(this.speechBubbles)) {
      if (bubble.container && bubble.sprite) {
        bubble.container.x = bubble.sprite.x;
        bubble.container.y = bubble.sprite.y - 20;
      }
    }
  }

  clearSpeechBubble(patientId) {
    const bubble = this.speechBubbles[patientId];
    if (bubble) {
      if (bubble.container) bubble.container.destroy();
      if (bubble.bg) bubble.bg.destroy();
      if (bubble.label) bubble.label.destroy();
      delete this.speechBubbles[patientId];
    }
    if (this.bubbleTimers[patientId]) {
      this.bubbleTimers[patientId].remove();
      delete this.bubbleTimers[patientId];
    }
  }

  clearAllBubbles() {
    Object.keys(this.speechBubbles).forEach(pid => this.clearSpeechBubble(pid));
  }

  updateBubbles(patients) {
    this.clearAllBubbles();
    if (this._hospBubbleBg) { this._hospBubbleBg.destroy(); this._hospBubbleBg = null; }
    if (this._hospBubbleLabel) { this._hospBubbleLabel.destroy(); this._hospBubbleLabel = null; }
    const alertPatients = [];
    const hospAEs = []; // collect AEs from hospital patients

    patients.forEach((p, i) => {
      const loc = (p.location || '').toUpperCase();
      if (loc === 'DECEASED') return;
      const ds = (p.disposition || p.DS?.DSDECOD || '').toUpperCase();
      if (ds.includes('DEATH') || ds.includes('DECEASED') || ds.includes('DIED')) return;

      const isHosp = loc === 'HOSPITAL' || loc === 'OUTPATIENT' || loc === 'INPATIENT';
      const text = _buildBubbleText(p);

      if (isHosp) {
        // Collect for hospital summary bubble
        if (text) hospAEs.push(text);
      } else if (text) {
        const maxGrade = Math.max(...(p.active_aes || []).map(ae => ae.grade || 0));
        const delay = i * 300 + Math.random() * 500;
        if (this._scene) {
          this._scene.time.delayedCall(delay, () => {
            this.showSpeechBubble(p.patient_id, text,
              maxGrade >= 4 ? 8000 : maxGrade >= 3 ? 6000 : 4000);
          });
        }
      }

      if (p.care_record && p.care_record.severity_level) {
        const sev = p.care_record.severity_level.toLowerCase();
        if (sev === 'red' || sev === 'orange') {
          alertPatients.push({
            patient_id: p.patient_id, severity: sev,
            summary: p.care_record.summary || 'Needs attention',
            actions: p.care_record.actions || [],
          });
        }
      }
    });

    // Hospital summary bubble
    if (hospAEs.length > 0 && this._scene && this.hospitalPos) {
      const maxShow = 4;
      let lines = hospAEs.slice(0, maxShow);
      if (hospAEs.length > maxShow) lines.push(`+${hospAEs.length - maxShow} more`);
      const summaryText = lines.join('\n');
      const bx = this.hospitalPos.x * TILE + TILE / 2;
      const by = this.hospitalPos.y * TILE - 40 - lines.length * 5;
      const tempText = this._scene.add.text(0, 0, summaryText, {
        fontFamily: '"NeoDunggeunmo", monospace', fontSize: '8px', color: '#fff', resolution: 4
      });
      const tw = tempText.width, th = tempText.height;
      tempText.destroy();
      const pad = 5;
      const bgW = tw + pad * 2, bgH = th + pad * 2;
      const bg = this._scene.add.graphics();
      bg.fillStyle(0x1b1f23, 0.92);
      bg.lineStyle(1, 0xf0883e, 0.7);
      bg.fillRoundedRect(bx - bgW / 2, by - bgH / 2, bgW, bgH, 4);
      bg.strokeRoundedRect(bx - bgW / 2, by - bgH / 2, bgW, bgH, 4);
      bg.fillStyle(0x1b1f23, 0.92);
      bg.fillTriangle(bx - 3, by + bgH / 2, bx + 3, by + bgH / 2, bx, by + bgH / 2 + 5);
      bg.setDepth(50);
      const label = this._scene.add.text(bx, by, summaryText, {
        fontFamily: '"NeoDunggeunmo", monospace', fontSize: '8px', color: '#e6edf3',
        align: 'center', resolution: 4,
      });
      label.setOrigin(0.5, 0.5).setDepth(51);
      bg.setAlpha(0); label.setAlpha(0);
      this._scene.tweens.add({ targets: [bg, label], alpha: 1, duration: 300 });
      this._hospBubbleBg = bg;
      this._hospBubbleLabel = label;
      this._scene.time.delayedCall(6000, () => {
        if (this._hospBubbleBg) { this._hospBubbleBg.destroy(); this._hospBubbleBg = null; }
        if (this._hospBubbleLabel) { this._hospBubbleLabel.destroy(); this._hospBubbleLabel = null; }
      });
    }

    if (alertPatients.length > 0 && this.onPhoneCall) {
      alertPatients.sort((a, b) => {
        const order = { red: 0, orange: 1 };
        return (order[a.severity] || 2) - (order[b.severity] || 2);
      });
      this.onPhoneCall(alertPatients);
    }
  }
}
