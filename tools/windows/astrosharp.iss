; Inno Setup 6 script for the Windows (ARM64) installer. Built by tools\package_windows.ps1,
; which passes AppVersion and SourceDir (the staged executables) with /D.

#ifndef AppVersion
  #error Define AppVersion (/DAppVersion=x.y.z); build with tools\package_windows.ps1
#endif
#ifndef SourceDir
  #error Define SourceDir (/DSourceDir=...); build with tools\package_windows.ps1
#endif

[Setup]
; Keep AppId unchanged: Windows uses it to recognise upgrades and the uninstall entry.
AppId={{3E567FCC-6090-4215-B228-CA7766945110}
AppName=AstroSharp
AppVersion={#AppVersion}
VersionInfoVersion={#AppVersion}
; Per-user install, no administrator rights: {autopf} is %LOCALAPPDATA%\Programs.
PrivilegesRequired=lowest
DefaultDirName={autopf}\AstroSharp
DisableProgramGroupPage=yes
ArchitecturesAllowed=arm64
ArchitecturesInstallIn64BitMode=arm64
OutputBaseFilename=AstroSharp-{#AppVersion}-arm64-setup
SetupIconFile=..\..\assets\AppIcon.ico
UninstallDisplayIcon={app}\AstroSharp.exe
UninstallDisplayName=AstroSharp
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
; The optional PATH entry needs running programs to be told about it.
ChangesEnvironment=yes

[Tasks]
Name: desktopicon; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: addtopath; Description: "Add astrosharp-cli to my PATH"; GroupDescription: "Command line:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\AstroSharp.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\astrosharp-cli.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\AstroSharp"; Filename: "{app}\AstroSharp.exe"
Name: "{autodesktop}\AstroSharp"; Filename: "{app}\AstroSharp.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AstroSharp.exe"; Description: "{cm:LaunchProgram,AstroSharp}"; Flags: nowait postinstall skipifsilent

[Code]
const
  EnvKey = 'Environment';

{ Position of Dir in the user's PATH (as in ';' + Path + ';'), or 0 if it isn't there. }
function PathPos(Dir: string; var Paths: string): Integer;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, EnvKey, 'Path', Paths) then
    Paths := '';
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';');
end;

{ Appends Dir, keeping a trailing separator if the PATH had one, so that RemoveFromPath
  restores the original value exactly. }
procedure AddToPath(Dir: string);
var
  Paths: string;
begin
  if PathPos(Dir, Paths) > 0 then
    exit;
  if (Paths = '') or (Paths[Length(Paths)] = ';') then
    Paths := Paths + Dir + ';'
  else
    Paths := Paths + ';' + Dir;
  RegWriteExpandStringValue(HKEY_CURRENT_USER, EnvKey, 'Path', Paths);
end;

procedure RemoveFromPath(Dir: string);
var
  Paths: string;
  P: Integer;
begin
  P := PathPos(Dir, Paths);
  if P = 0 then
    exit;
  { P is where Dir starts in Paths. Remove it with the separator after it, or, when it is
    the last entry, with the one before it. }
  if P + Length(Dir) <= Length(Paths) then
    Delete(Paths, P, Length(Dir) + 1)
  else if P > 1 then
    Delete(Paths, P - 1, Length(Dir) + 1)
  else
    Delete(Paths, 1, Length(Dir));
  RegWriteExpandStringValue(HKEY_CURRENT_USER, EnvKey, 'Path', Paths);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addtopath') then
    AddToPath(ExpandConstant('{app}'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromPath(ExpandConstant('{app}'));
end;
