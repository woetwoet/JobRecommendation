param([string]$docpath,[string]$htmlpath = $docpath)

$wdTypes = Add-Type -AssemblyName 'Microsoft.Office.Interop.Word' -Passthru
$wdSaveFormat = $wdTypes | Where {$_.Name -eq "wdSaveFormat"}
Write-Host "Start"

#$srcfiles = Get-ChildItem -Path $docPath -Recurse -filter "*.doc*" 
$srcfiles = Get-ChildItem -Path $docPath -Recurse -filter "*.pdf*" 
$saveFormat = [Enum]::Parse([Microsoft.Office.Interop.Word.WdSaveFormat], "wdFormatText");
$word = new-object -comobject word.application
$word.Visible = $False
$archive = "C:\Users\woutv\Jupyter\Jobs\inputdata\Processed\"

		
function saveas-filteredtxt($filename)
	{
		$opendoc = $word.documents.open($doc.FullName);
		$opendoc.saveas([ref]"$htmlpath\$filename.txt", [ref]$saveFormat);
		$opendoc.close();
		
	}
	
ForEach ($doc in $srcfiles)
	{
		Write-Host "Processing :" $doc.FullName
		$tempfilename = $doc.FullName -replace "C:\\Users\\woutv\\Jupyter\\Jobs\\inputdata\\cvsraw\\" , ""
		$newfilename = $tempfilename -replace "\\" , "_"
		saveas-filteredtxt($newfilename)
		Write-Host "Move file "$newfilename
		Move-Item -Path $doc.FullName -Destination $archive$newfilename	
		$doc = $null
		$newfilename = $null
		
	}

$word.quit();