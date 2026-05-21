param(
    [Parameter(Mandatory = $true)]
    [string]$LabName
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourcePython = Join-Path $projectRoot "imitation_lab.py"
$sourceMarkdown = Join-Path $projectRoot "imitation_lab.md"
$targetPython = Join-Path $projectRoot "$LabName.py"
$targetMarkdown = Join-Path $projectRoot "$LabName.md"
$labJsonPath = Join-Path $projectRoot "lab.json"

if (-not (Test-Path -LiteralPath $sourcePython)) {
    throw "Missing source scene entrypoint: $sourcePython"
}

if (-not (Test-Path -LiteralPath $sourceMarkdown)) {
    throw "Missing source markdown file: $sourceMarkdown"
}

Move-Item -LiteralPath $sourcePython -Destination $targetPython
Move-Item -LiteralPath $sourceMarkdown -Destination $targetMarkdown

$markdown = Get-Content -LiteralPath $targetMarkdown -Raw
$markdown = $markdown.Replace("Emio Imitation Learning", $LabName)
Set-Content -LiteralPath $targetMarkdown -Value $markdown -Encoding UTF8

$labJson = Get-Content -LiteralPath $labJsonPath -Raw
$labJson = $labJson.Replace('"name": "imitation lab"', ('"name": "{0}"' -f $LabName))
$labJson = $labJson.Replace('"filename": "imitation_lab.md"', ('"filename": "{0}.md"' -f $LabName))
$labJson = $labJson.Replace('"title": "Imitation Lab"', ('"title": "{0}"' -f $LabName))
Set-Content -LiteralPath $labJsonPath -Value $labJson -Encoding UTF8

Write-Host "Done renaming lab: '$LabName'"
