// export_stats.js
const ee        = require('@google/earthengine');
const {JWT}     = require('google-auth-library');
const fs        = require('fs-extra');
const { parse } = require('csv-parse/sync');
const pLimitMod = require('p-limit');
const pLimit    = pLimitMod.default || pLimitMod;  // handle default export

// ------------ 1) AUTHENTICATE via Service Account ------------
const saKey = fs.readJsonSync('./service-account.json');
ee.data.authenticateViaPrivateKey(
  {
    client_email: saKey.client_email,
    private_key:  saKey.private_key
  },
  () => {
    console.log(' Authenticated with service account');
    ee.initialize(null, null,
      () => {
        console.log(' Earth Engine initialized');
        runAllJobs();
      },
      err => console.error(' EE init error:', err)
    );
  },
  err => console.error(' Auth error:', err)
);

// ------------ 2) PROMISIFIED getInfo() ------------
function getInfoPromise(fc) {
  return new Promise((resolve, reject) =>
    fc.getInfo(res => res.error ? reject(res.error) : resolve(res))
  );
}

// ------------ 3) STATS FACTORY ------------
function seasonStatsFactory(start, end, season, year) {
  return feat => {
    const geom = feat.geometry();
    const col  = ee.ImageCollection('MODIS/061/MOD13A1')
                   .filterDate(start, end)
                   .select('EVI');
    // Basic stats
    const statsImg = ee.Image.cat([
      col.mean().rename('mean'),
      col.max().rename('max'),
      col.min().rename('min'),
      col.reduce(ee.Reducer.stdDev()).rename('stdDev'),
      col.reduce(ee.Reducer.percentile([90])).rename('p90')
    ]);
    const summary = statsImg.reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: geom, scale:500, bestEffort:true
    });

    // Linear trend
    const addTime = img => {
      const days = ee.Number(img.get('system:time_start'))
                     .divide(1000*60*60*24);
      return img.addBands(
        ee.Image.constant(0).add(days)
               .toFloat().rename('time')
      );
    };
    const linStats = col
      .map(addTime)
      .select(['time','EVI'])
      .reduce(ee.Reducer.linearFit())
      .reduceRegion({
        reducer: ee.Reducer.first(),
        geometry: geom, scale:500, bestEffort:true
      });

    // AUC
    const sumDict = col
      .sum()
      .reduceRegion({
        reducer: ee.Reducer.first(),
        geometry: geom, scale:500, bestEffort:true
      });
    const auc = ee.Number(sumDict.get('EVI')).multiply(16);

    return feat.set({
      season, year,
      mean:      summary.get('mean'),
      max:       summary.get('max'),
      min:       summary.get('min'),
      stdDev:    summary.get('stdDev'),
      p90:       summary.get('p90'),
      slope:     linStats.get('scale'),
      intercept: linStats.get('offset'),
      auc
    });
  };
}

// ------------ 4) MAIN RUNNER ------------
async function runAllJobs() {
  // Load coords
  const raw     = fs.readFileSync('loc_coords_checkpoint.csv', 'utf8');
  const records = parse(raw, { columns: true, skip_empty_lines: true });
  const allFeats = records
    .map(r => {
      const lat = parseFloat(r.lat), lon = parseFloat(r.lon);
      if (isNaN(lat)||isNaN(lon)) return null;
      const id = r.id || `${r.VILLAGE}_${r.Zipcode}`;
      return ee.Feature(ee.Geometry.Point([lon, lat]), { id });
    })
    .filter(Boolean);

  const totalPoints = allFeats.length;
  const targetCalls = 900;
  const chunkSize   = Math.ceil(totalPoints / targetCalls);
  const nChunks     = Math.ceil(totalPoints / chunkSize);

  console.log(`Total points: ${totalPoints}`);
  console.log(` → chunkSize: ${chunkSize}, chunks/job: ${nChunks}`);

  // Build season×year jobs
  const seasons = ['Kharif','Rabi'];
  const years   = [2020,2021,2022];
  const jobs    = [];
  years.forEach(year =>
    seasons.forEach(season => {
      const start = season==='Kharif'
        ? `${year}-06-01`
        : `${year}-11-01`;
      const end   = season==='Kharif'
        ? `${year}-10-31`
        : `${year+1}-04-30`;
      jobs.push({ season, year, start, end });
    })
  );

  console.log(`\n→ Running ${jobs.length} jobs in parallel (up to 6 at once)\n`);
  const limiter = pLimit(6);

  // Launch all jobs
  await Promise.all(
    jobs.map(job => limiter(() => processJob(job, allFeats, chunkSize, nChunks)))
  );

  console.log('\n All season×year CSVs generated!');
}

// ------------ 5) PROCESS SINGLE JOB ------------
async function processJob(job, allFeats, chunkSize, nChunks) {
  const { season, year, start, end } = job;
  const outName = `stats_${season}_${year}.csv`;

  // Write header
  const header = [
    'id','season','year',
    'mean','max','min','stdDev','p90',
    'slope','intercept','auc'
  ].join(',') + '\n';
  fs.writeFileSync(outName, header);

  console.log(`=== Job: ${season} ${year} (${start}→${end}) ===`);
  for (let i = 0; i < allFeats.length; i += chunkSize) {
    const chunkIdx = i / chunkSize + 1;
    console.log(`  [${season} ${year}] chunk ${chunkIdx}/${nChunks}`);
    const batch = allFeats.slice(i, i + chunkSize);
    const fc = ee.FeatureCollection(batch)
                 .map(seasonStatsFactory(start, end, season, year));
    const info = await getInfoPromise(fc);

    // Append data
    const lines = info.features
      .map(f => {
        const p = f.properties;
        return [
          p.id, season, year,
          p.mean, p.max, p.min, p.stdDev, p.p90,
          p.slope, p.intercept, p.auc
        ].join(',');
      })
      .join('\n') + '\n';
    fs.appendFileSync(outName, lines);
  }

  console.log(` Completed ${outName} (${allFeats.length} pts in ${nChunks} calls)`);
}