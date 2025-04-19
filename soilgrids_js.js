// export_soil_sixlayers.js
const ee        = require('@google/earthengine');
const {JWT}     = require('google-auth-library');
const fs        = require('fs-extra');
const { parse } = require('csv-parse/sync');
const pLimitMod = require('p-limit');
const pLimit    = pLimitMod.default || pLimitMod;
const path      = require('path');

// CONFIG
const BANDS     = [
  'bdod_0-5cm_mean',
  'bdod_5-15cm_mean',
  'bdod_15-30cm_mean',
  'bdod_30-60cm_mean',
  'bdod_60-100cm_mean',
  'bdod_100-200cm_mean'
];
const CONCURRENCY = 40;
const OUT_CSV     = 'soilgrids_sixlayers.csv';
const CKPT_FILE   = 'soilgrids_sixlayers_checkpoint.json';

// 1) AUTH
const key = fs.readJsonSync(path.join(__dirname,'service-account.json'));
ee.data.authenticateViaPrivateKey(
  { client_email: key.client_email, private_key: key.private_key },
  () => {
    console.log(' Authenticated');
    ee.initialize(null, null, runSampling, err => console.error(' EE init error:', err));
  },
  err => console.error(' Auth error:', err)
);

// 2) PROMISIFIED evaluate()
function evaluatePromise(obj) {
  return new Promise((resolve, reject) => {
    obj.evaluate((res, err) => {
      if (err) reject(err);
      else resolve(res);
    });
  });
}

async function runSampling() {
  // 3) LOAD POINTS
  const raw     = fs.readFileSync('loc_coords_checkpoint.csv','utf8');
  const records = parse(raw, { columns:true, skip_empty_lines:true });
  const pts = records
    .map((r,i) => {
      const lat = +r.lat, lon = +r.lon;
      if (isNaN(lat)||isNaN(lon)) return null;
      return { id: (r.id||String(i)), lat, lon };
    })
    .filter(Boolean);
  console.log(`Loaded ${pts.length} points`);

  // 4) CHUNKING
  const chunkSize = Math.ceil(pts.length / CONCURRENCY);
  const chunks    = [];
  for (let i=0; i<pts.length; i+=chunkSize) {
    chunks.push(pts.slice(i, i+chunkSize));
  }
  console.log(`→ ${chunks.length} chunks (~${chunkSize} pts each)`);

  // 5) CHECKPOINTING
  let ckpt = { done: [] };
  if (fs.existsSync(CKPT_FILE)) {
    ckpt = fs.readJsonSync(CKPT_FILE);
    console.log(`Resuming: ${ckpt.done.length}/${chunks.length} chunks done`);
  } else {
    // fresh run → write CSV header
    const header = ['id', ...BANDS].join(',') + '\n';
    fs.writeFileSync(OUT_CSV, header);
    console.log(`Starting fresh; header written to ${OUT_CSV}`);
  }

  // 6) EE IMAGE: select all six bands
  const rawImg = ee.Image(`projects/soilgrids-isric/bdod_mean`);
  const soilImg = rawImg.select(BANDS);

  // 7) PARALLEL SAMPLE
  const limit = pLimit(CONCURRENCY);
  await Promise.all(
    chunks.map((chunk, ci) => limit(async () => {
      if (ckpt.done.includes(ci)) {
        console.log(`[Skip ${ci+1}/${chunks.length}]`);
        return;
      }
      console.log(`[Chunk ${ci+1}/${chunks.length}] sampling ${chunk.length} pts`);

      const fc = ee.FeatureCollection(
        chunk.map(p => ee.Feature(ee.Geometry.Point([p.lon,p.lat]), { id: p.id }))
      );
      const sampled = soilImg.sampleRegions({
        collection: fc,
        scale:      250,
        properties: ['id'],
        geometries: false
      });

      try {
        const info = await evaluatePromise(sampled);
        // build CSV lines
        const lines = info.features.map(f => {
          const props = f.properties;
          return [props.id, ...BANDS.map(b => props[b])].join(',');
        }).join('\n') + '\n';

        fs.appendFileSync(OUT_CSV, lines);
        ckpt.done.push(ci);
        fs.writeJsonSync(CKPT_FILE, ckpt);
        console.log(`[Chunk ${ci+1}/${chunks.length}] done & checkpointed`);
      } catch (err) {
        console.error(`[Chunk ${ci+1}/${chunks.length}] ERROR:`, err);
      }
    }))
  );

  console.log(`\n All chunks complete. Output in '${OUT_CSV}'`);
}
