# Start a command as a process that outlives the shell that launched it.
# Win32_Process.Create places the process outside the caller's job object, so closing
# the terminal or the session does not stop it. Output goes to the given log file.
#   powershell -File scripts/launch_detached.ps1 -Command "python -u scripts/pipeline.py" -Log C:\path\pipeline.out
param(
    [Parameter(Mandatory = $true)][string]$Command,
    [Parameter(Mandatory = $true)][string]$Log,
    [string]$WorkingDirectory = (Split-Path -Parent $PSScriptRoot)
)
$line = "cmd.exe /c `"$Command > `"$Log`" 2>&1`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $line
    CurrentDirectory = $WorkingDirectory
}
if ($result.ReturnValue -ne 0) { throw "Win32_Process.Create failed with $($result.ReturnValue)" }
"started process $($result.ProcessId)"
