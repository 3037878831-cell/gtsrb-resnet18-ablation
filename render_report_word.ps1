param(
    [string]$InputDocx = "",
    [string]$OutputDir = "$PSScriptRoot\reports\render"
)

$ErrorActionPreference = "Stop"
if (-not $InputDocx) {
    $InputDocx = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot "reports") -Filter "*.docx" |
        Select-Object -First 1 -ExpandProperty FullName
}
$InputDocx = [System.IO.Path]::GetFullPath($InputDocx)
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$PdfPath = Join-Path $OutputDir "report.pdf"

$Word = New-Object -ComObject Word.Application
$Word.Visible = $false
$Word.DisplayAlerts = 0
try {
    Write-Host "Opening DOCX"
    $Document = $Word.Documents.Open($InputDocx, $false, $true)
    try {
        Write-Host "DOCX opened"
        $PageCount = $Document.ComputeStatistics(2)
        Write-Host "Pages: $PageCount"
        Write-Host "Saving PDF"
        $Word.ActivePrinter = "Microsoft Print to PDF"
        $Document.SaveAs2($PdfPath, 17)
        Write-Host "PDF saved"
    } finally {
        $Document.Close($false)
    }
} finally {
    $Word.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($Word) | Out-Null
}

$Pdftoppm = "C:\Users\Aurora\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe"
& $Pdftoppm -png -r 130 $PdfPath (Join-Path $OutputDir "page")
if ($LASTEXITCODE -ne 0) { throw "pdftoppm failed" }
Get-ChildItem -LiteralPath $OutputDir -Filter "page-*.png" | Sort-Object Name | Select-Object Name, Length
