$props  = @('soc','phh2o','nitrogen','cfvo','cec','sand','silt','clay','bdod')
$depths = @('0-5cm','5-15cm','15-30cm','30-60cm','60-100cm','100-200cm')
$base   = 'https://files.isric.org/soilgrids/latest/data'
$root   = 'soilgrids_raw'

# 2) Create output directory
if (-not (Test-Path $root)) {
    New-Item -ItemType Directory -Path $root | Out-Null
}

# 3) Download all depths for each property
foreach ($prop in $props) {
    Write-Host "→ Downloading depths for '$prop'"
    $propDir = Join-Path $root $prop
    if (-not (Test-Path $propDir)) {
        New-Item -ItemType Directory -Path $propDir | Out-Null
    }
    foreach ($d in $depths) {
        $url = "$base/$prop/${prop}_${d}_mean.vrt"
        $out = Join-Path $propDir "${prop}_${d}_mean.vrt"
        Write-Host "   • $url"
        Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
    }
}

# 4) Download WRB top layer (0–5 cm)
Write-Host "→ Downloading WRB most-probable (0-5 cm)"
$wrbDir = Join-Path $root 'wrb'
if (-not (Test-Path $wrbDir)) {
    New-Item -ItemType Directory -Path $wrbDir | Out-Null
}
$wrbUrl = "$base/wrb/wrb_0-5cm.vrt"
$wrbOut = Join-Path $wrbDir 'wrb_0-5cm.vrt'
Write-Host "   • $wrbUrl"
Invoke-WebRequest -Uri $wrbUrl -OutFile $wrbOut -UseBasicParsing

Write-Host "`nAll VRTs downloaded into '$root\' (you can delete this folder after parsing)"
