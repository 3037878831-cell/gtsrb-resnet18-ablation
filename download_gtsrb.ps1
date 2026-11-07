param(
    [int]$Connections = 24
)

$ErrorActionPreference = "Stop"
$Url = "https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip"
$ExpectedLength = 123524425L
$ProjectRoot = $PSScriptRoot
$DataRoot = Join-Path $ProjectRoot "data"
$ArchivePath = Join-Path $DataRoot "traffic-signs-data.zip"
$PartsRoot = Join-Path $DataRoot "download-parts"
$ExtractRoot = Join-Path $DataRoot "traffic-signs-data"

New-Item -ItemType Directory -Force -Path $DataRoot, $PartsRoot | Out-Null

if ((Test-Path -LiteralPath $ArchivePath) -and
    ((Get-Item -LiteralPath $ArchivePath).Length -eq $ExpectedLength)) {
    Write-Host "Archive already downloaded."
} else {
    $ChunkSize = [math]::Ceiling($ExpectedLength / $Connections)
    $Jobs = for ($Index = 0; $Index -lt $Connections; $Index++) {
        $Start = [long]$Index * $ChunkSize
        $End = [math]::Min($ExpectedLength - 1, $Start + $ChunkSize - 1)
        $PartPath = Join-Path $PartsRoot ("part-{0:D3}.bin" -f $Index)
        $ExpectedPartLength = $End - $Start + 1
        Start-Job -ScriptBlock {
            param($Url, $Start, $End, $PartPath, $ExpectedPartLength)
            if ((Test-Path -LiteralPath $PartPath) -and
                ((Get-Item -LiteralPath $PartPath).Length -eq $ExpectedPartLength)) {
                return "cached $PartPath"
            }
            $ExistingLength = if (Test-Path -LiteralPath $PartPath) {
                (Get-Item -LiteralPath $PartPath).Length
            } else { 0L }
            if ($ExistingLength -gt $ExpectedPartLength) {
                Remove-Item -LiteralPath $PartPath -Force
                $ExistingLength = 0L
            }
            $ResumeStart = $Start + $ExistingLength
            $RemainingPath = "$PartPath.remaining"
            Remove-Item -LiteralPath $RemainingPath -Force -ErrorAction SilentlyContinue
            & curl.exe -sS -L --fail --retry 20 --retry-all-errors `
                --connect-timeout 30 --range "$ResumeStart-$End" --output $RemainingPath $Url
            if ($LASTEXITCODE -ne 0) {
                throw "curl failed for byte range $ResumeStart-$End"
            }
            $OutputStream = [System.IO.File]::Open(
                $PartPath, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write
            )
            $InputStream = [System.IO.File]::OpenRead($RemainingPath)
            try { $InputStream.CopyTo($OutputStream) } finally {
                $InputStream.Dispose()
                $OutputStream.Dispose()
            }
            Remove-Item -LiteralPath $RemainingPath -Force
            $ActualLength = (Get-Item -LiteralPath $PartPath).Length
            if ($ActualLength -ne $ExpectedPartLength) {
                throw "Part length mismatch: expected $ExpectedPartLength, got $ActualLength"
            }
            return "downloaded $PartPath"
        } -ArgumentList $Url, $Start, $End, $PartPath, $ExpectedPartLength
    }

    while (($Jobs | Where-Object State -in "Running", "NotStarted").Count -gt 0) {
        $Done = ($Jobs | Where-Object State -eq "Completed").Count
        $Failed = ($Jobs | Where-Object State -eq "Failed").Count
        Write-Host "Chunks completed: $Done/$Connections; failed: $Failed"
        Start-Sleep -Seconds 10
    }
    $Jobs | Receive-Job
    $FailedJobs = $Jobs | Where-Object State -ne "Completed"
    if ($FailedJobs) {
        $FailedJobs | Format-List Id, State, ChildJobs
        throw "One or more download chunks failed. Run this script again to resume."
    }
    $Jobs | Remove-Job

    $Output = [System.IO.File]::Open($ArchivePath, [System.IO.FileMode]::Create)
    try {
        Get-ChildItem -LiteralPath $PartsRoot -Filter "part-*.bin" |
            Sort-Object Name |
            ForEach-Object {
                $Input = [System.IO.File]::OpenRead($_.FullName)
                try { $Input.CopyTo($Output) } finally { $Input.Dispose() }
            }
    } finally {
        $Output.Dispose()
    }
    if ((Get-Item -LiteralPath $ArchivePath).Length -ne $ExpectedLength) {
        throw "Final archive length is incorrect."
    }
}

& "$ProjectRoot\..\.conda\gtsrb\python.exe" -c `
    "import zipfile; p=r'$ArchivePath'; z=zipfile.ZipFile(p); bad=z.testzip(); assert bad is None, bad; print('ZIP validation passed:', len(z.namelist()), 'files')"
if ($LASTEXITCODE -ne 0) { throw "ZIP validation failed." }

if (-not (Test-Path -LiteralPath (Join-Path $ExtractRoot "train.p"))) {
    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $ExtractRoot -Force
}
Get-ChildItem -LiteralPath $ExtractRoot | Select-Object Name, Length
