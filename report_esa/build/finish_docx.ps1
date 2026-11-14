# Finish the pandoc output with Word: guideline formatting (A4, 2 cm margins, Times New Roman, sizes, 1.5 spacing), one section per
# chapter with the header/footer rules, table formatting, page numbers for the contents lists. Run:
#   powershell -File build\finish_docx.ps1 -In <raw.docx> -Out <report.docx> -Pdf <report.pdf> -Meta <meta.json>
param([string]$In, [string]$Out, [string]$Pdf, [string]$Meta)
$ErrorActionPreference = "Stop"
$metaObj = Get-Content -Raw -Encoding UTF8 $Meta | ConvertFrom-Json
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($In)

  function Set-Sty($name, $size, $bold, $align, $before, $after, $single, $italic) {
    try { $s = $doc.Styles.Item($name) } catch { Write-Host "style missing: $name"; return }
    $s.Font.Name = "Times New Roman"; $s.Font.Size = $size; $s.Font.Bold = $bold; $s.Font.Italic = $italic; $s.Font.Color = 0
    $pf = $s.ParagraphFormat
    $pf.Alignment = $align; $pf.SpaceBefore = $before; $pf.SpaceAfter = $after
    if ($single) { $pf.LineSpacingRule = 0 } else { $pf.LineSpacingRule = 5; $pf.LineSpacing = 18 }
  }
  # body 12 justified 1.5; headings per guideline: chapter 18 centred, section 16 bold left, subsection 14 bold left
  foreach ($n in "Normal", "Body Text", "First Paragraph", "Compact", "Block Text") { Set-Sty $n 12 0 3 0 6 $false 0 }
  Set-Sty "Heading 1" 18 1 1 0 12 $false 0
  Set-Sty "Heading 2" 16 1 0 12 6 $false 0
  Set-Sty "Heading 3" 14 1 0 10 4 $false 0
  Set-Sty "ChapterNo" 18 1 1 0 6 $false 0
  Set-Sty "FrontHeading" 18 1 1 0 18 $false 0
  Set-Sty "TitleCenter" 14 1 1 6 6 $true 0
  Set-Sty "CenterText" 12 0 1 6 6 $true 0
  Set-Sty "Equation" 12 0 1 6 6 $true 0
  Set-Sty "Table Caption" 11 1 1 8 4 $true 0
  Set-Sty "Image Caption" 11 1 1 4 10 $true 0
  $doc.Styles.Item("Heading 1").ParagraphFormat.KeepWithNext = -1
  $doc.Styles.Item("Heading 2").ParagraphFormat.KeepWithNext = -1
  $doc.Styles.Item("Heading 3").ParagraphFormat.KeepWithNext = -1
  $doc.Styles.Item("Table Caption").ParagraphFormat.KeepWithNext = -1
  $doc.Styles.Item("FrontHeading").ParagraphFormat.PageBreakBefore = -1
  $doc.Styles.Item("ChapterNo").ParagraphFormat.KeepWithNext = -1
  try { $c = $doc.Styles.Item("Placeholder"); $c.Font.Color = 255; $c.Font.Bold = 1 } catch { Write-Host "no Placeholder style" }
  try { $c = $doc.Styles.Item("Verbatim Char"); $c.Font.Name = "Courier New"; $c.Font.Size = 10 } catch { Write-Host "no Verbatim Char" }

  # page setup: A4, 2 cm margins
  $cm = 28.3465
  foreach ($sec in $doc.Sections) {
    $ps = $sec.PageSetup
    $ps.PaperSize = 7
    $ps.TopMargin = 2 * $cm; $ps.BottomMargin = 2 * $cm; $ps.LeftMargin = 2 * $cm; $ps.RightMargin = 2 * $cm
    $ps.HeaderDistance = 1 * $cm; $ps.FooterDistance = 1 * $cm
  }

  # one section per chapter (new page), inserted before each ChapterNo paragraph
  $idx = @()
  $i = 0
  foreach ($p in $doc.Paragraphs) { $i++; if ($p.Style.NameLocal -eq "ChapterNo") { $idx += $i } }
  [array]::Reverse($idx)
  foreach ($k in $idx) { $r = $doc.Paragraphs.Item($k).Range; $r.Collapse(1); $r.InsertBreak(2) }
  Write-Host ("sections: " + $doc.Sections.Count)

  # headers and footers: none in the front matter; from the Introduction on: header on every page except a chapter's first page,
  # footer (page number) only on a chapter's first page; numbering restarts at 1 at the Introduction
  for ($s = 1; $s -le $doc.Sections.Count; $s++) {
    $sec = $doc.Sections.Item($s)
    if ($s -eq 1) {
      $sec.PageSetup.DifferentFirstPageHeaderFooter = 0
      foreach ($kind in 1, 2, 3) { $sec.Headers.Item($kind).Range.Text = ""; $sec.Footers.Item($kind).Range.Text = "" }
      continue
    }
    $sec.PageSetup.DifferentFirstPageHeaderFooter = -1
    foreach ($kind in 1, 2) { $sec.Headers.Item($kind).LinkToPrevious = $false; $sec.Footers.Item($kind).LinkToPrevious = $false }
    $h = $sec.Headers.Item(1)
    $h.Range.Text = "UE24CS320A Capstone Project Phase - 1`t" + $metaObj.short
    $h.Range.Font.Name = "Times New Roman"; $h.Range.Font.Size = 10
    $h.Range.ParagraphFormat.TabStops.ClearAll()
    [void]$h.Range.ParagraphFormat.TabStops.Add(17 * $cm, 2)
    $h.Range.ParagraphFormat.Alignment = 0
    $h.Range.ParagraphFormat.SpaceAfter = 0; $h.Range.ParagraphFormat.LineSpacingRule = 0
    $h.Range.ParagraphFormat.Borders.Item(-3).LineStyle = 1
    $sec.Headers.Item(2).Range.Text = ""
    $sec.Footers.Item(1).Range.Text = ""
    $f = $sec.Footers.Item(2)
    $f.Range.Text = ""
    $f.Range.ParagraphFormat.Alignment = 1
    [void]$f.Range.Fields.Add($f.Range, 33)
    $f.Range.Font.Name = "Times New Roman"; $f.Range.Font.Size = 12
    if ($s -eq 2) { $pn = $sec.Footers.Item(1).PageNumbers; $pn.RestartNumberingAtSection = $true; $pn.StartingNumber = 1 }
  }

  # tables
  $nFrontPlain = [int]$metaObj.front_plain_tables
  for ($t = 1; $t -le $doc.Tables.Count; $t++) {
    $tb = $doc.Tables.Item($t)
    if ($t -le $nFrontPlain) {
      if ($tb.Rows.Item(1).Range.Text.Trim().Length -le 6 -and $tb.Rows.Count -gt 1) { $tb.Rows.Item(1).Delete() }
      $tb.Borders.Enable = $false
      $tb.AllowAutoFit = $true; $tb.PreferredWidthType = 2; $tb.PreferredWidth = 100; $tb.AutoFitBehavior(2)
      $tb.Range.Font.Size = 11
      $tb.Range.ParagraphFormat.LineSpacingRule = 0; $tb.Range.ParagraphFormat.Alignment = 1; $tb.Range.ParagraphFormat.SpaceAfter = 2
      continue
    }
    $tb.Borders.Enable = $true
    $tb.Range.Font.Size = $(if ($t -le $nFrontPlain + 3) { 11 } else { 10 })
    $tb.Range.ParagraphFormat.LineSpacingRule = 0; $tb.Range.ParagraphFormat.Alignment = 0
    $tb.Range.ParagraphFormat.SpaceBefore = 1; $tb.Range.ParagraphFormat.SpaceAfter = 2
    $tb.Rows.AllowBreakAcrossPages = 0
    $tb.Rows.Item(1).HeadingFormat = -1
    $tb.Rows.Item(1).Range.Font.Bold = 1
    $tb.Rows.Item(1).Shading.BackgroundPatternColor = 14277081
    $tb.PreferredWidthType = 2; $tb.PreferredWidth = 100
    if ($t -le $nFrontPlain + 3) {
      $tb.AllowAutoFit = $false
      $tb.Columns.Item(1).Width = 75; $tb.Columns.Item(3).Width = 55; $tb.Columns.Item(2).Width = 351
      for ($r = 1; $r -le $tb.Rows.Count; $r++) { $tb.Cell($r, 3).Range.ParagraphFormat.Alignment = 2 }
    } else {
      $tb.AllowAutoFit = $true
      $tb.AutoFitBehavior(1); $tb.AutoFitBehavior(2)
    }
  }

  # explicit widths for the certificate and declaration tables (pandoc's widths wrap "Signature")
  $t1 = $doc.Tables.Item(1); $t1.AllowAutoFit = $false
  $t1.Columns.Item(1).Width = 150; $t1.Columns.Item(2).Width = 170; $t1.Columns.Item(3).Width = 160
  $t2 = $doc.Tables.Item(2); $t2.AllowAutoFit = $false
  $t2.Columns.Item(1).Width = 90; $t2.Columns.Item(2).Width = 210; $t2.Columns.Item(3).Width = 180
  foreach ($ti in 4, 5) { $tq = $doc.Tables.Item($ti); $tq.AllowAutoFit = $false; $tq.Columns.Item(1).Width = 62; $tq.Columns.Item(2).Width = 358; $tq.Columns.Item(3).Width = 62 }

  # page numbers need a saved, reopened document: Information() returns 0 inside the first session
  $doc.SaveAs2($Out, 16); $doc.Close(0)
  $doc = $word.Documents.Open($Out)
  $doc.Repaginate()

  # page numbers of headings and captions (displayed numbers, restarted at the Introduction)
  $pages = @{}
  foreach ($p in $doc.Paragraphs) {
    $sn = $p.Style.NameLocal
    if ($sn -in "Heading 1", "Heading 2", "Heading 3", "Table Caption", "Image Caption") {
      $txt = ($p.Range.Text -replace "[\r\a]", "").Trim()
      if ($txt -and -not $pages.ContainsKey($txt)) { $pages[$txt] = $p.Range.Information(1) }
    }
  }
  Write-Host ("headings found: " + $pages.Count + "; sample: " + (($pages.GetEnumerator() | Select-Object -First 3 | ForEach-Object { $_.Key + "=" + $_.Value }) -join " | "))
  function Find-Page($key) {
    if ($pages.ContainsKey($key)) { return $pages[$key] }
    foreach ($k in $pages.Keys) { if ($k.StartsWith($key)) { return $pages[$k] } }
    return $null
  }
  $lists = @(@{rows=@($metaObj.toc); tbl=($nFrontPlain + 1)}, @{rows=@($metaObj.lot); tbl=($nFrontPlain + 2)}, @{rows=@($metaObj.lof); tbl=($nFrontPlain + 3)})
  Write-Host ("lists: " + $lists.Count + "; toc rows " + $lists[0].rows.Count)
  $missing = @()
  foreach ($L in $lists) {
    $rows = $L.rows; $tb = $doc.Tables.Item([int]$L.tbl)
    for ($r = 0; $r -lt $rows.Count; $r++) {
      $pg = Find-Page $rows[$r].key
      if ($null -eq $pg) { $missing += $rows[$r].key; $pg = "?" }
      if ($r -lt 2) { Write-Host ("write: " + $rows[$r].key + " -> '" + $pg + "' type " + $pg.GetType().Name + " rows=" + $tb.Rows.Count) }
      $tb.Cell($r + 2, 3).Range.Text = "$pg"
      $tb.Cell($r + 2, 3).Range.ParagraphFormat.Alignment = 2
    }
  }
  if ($missing.Count) { Write-Host ("page not found for: " + ($missing -join " | ")) }
  Write-Host ("total pages: " + $doc.ComputeStatistics(2))
  $doc.Save()
  $doc.ExportAsFixedFormat($Pdf, 17)
  $doc.Close(0)
} finally {
  $word.Quit()
}
