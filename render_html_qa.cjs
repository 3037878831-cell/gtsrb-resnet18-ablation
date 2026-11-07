const path = require("path");
const { pathToFileURL } = require("url");
const { chromium } = require("C:/Users/Aurora/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");

(async () => {
  const input = path.resolve(process.argv[2]);
  const output = path.resolve(process.argv[3]);
  const browser = await chromium.launch({
    executablePath: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    headless: true,
  });
  const page = await browser.newPage({ viewport: { width: 1275, height: 1650 } });
  await page.goto(pathToFileURL(input).href, { waitUntil: "networkidle" });
  await page.emulateMedia({ media: "print" });
  await page.pdf({
    path: output,
    format: "Letter",
    printBackground: true,
    margin: { top: "0.25in", right: "0.25in", bottom: "0.25in", left: "0.25in" },
    preferCSSPageSize: true,
  });
  await browser.close();
  process.stdout.write(output + "\n");
})().catch((error) => {
  process.stderr.write(error.stack + "\n");
  process.exit(1);
});

