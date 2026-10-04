const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { Resvg } = require("@resvg/resvg-js");

const source = path.join(__dirname, "node_modules/bootstrap-icons");
const output = path.resolve(__dirname, "../../data/symbols");
const version = JSON.parse(
  fs.readFileSync(path.join(source, "package.json"), "utf8")
).version;
const files = fs.readdirSync(path.join(source, "icons"))
  .filter(name => name.endsWith(".svg"))
  .sort();

const patterns = new Map();
let empty = 0;

for (const file of files) {
  const svg = fs.readFileSync(path.join(source, "icons", file), "utf8");
  const rendered = new Resvg(svg, {
    fitTo: { mode: "width", value: 16 },
    font: { loadSystemFonts: false }
  }).render();

  if (rendered.width !== 16 || rendered.height !== 16) {
    throw new Error(`Unexpected dimensions: ${file}`);
  }

  const rgba = rendered.pixels;
  if (rgba.length !== 16 * 16 * 4) {
    throw new Error(`Unexpected pixel buffer: ${file}`);
  }

  // Transparent background = 0; at least half-covered foreground = 1.
  const bits = Array.from({ length: 256 }, (_, i) =>
    rgba[i * 4 + 3] >= 128 ? 1 : 0
  );
  if (!bits.some(Boolean)) {
    empty++;
    continue;
  }

  const key = bits.join("");
  const name = file.slice(0, -4);
  if (patterns.has(key)) {
    patterns.get(key).names.push(name);
  } else {
    patterns.set(key, {
      id: crypto.createHash("sha256").update(key).digest("hex"),
      names: [name],
      bits
    });
  }
}

fs.mkdirSync(output, { recursive: true });
fs.copyFileSync(path.join(source, "LICENSE"), path.join(output, "LICENSE"));
fs.writeFileSync(path.join(output, "catalogue.json"), JSON.stringify({
  source: "bootstrap-icons",
  sourceVersion: version,
  rendererVersion: require("@resvg/resvg-js/package.json").version,
  width: 16,
  height: 16,
  alphaThreshold: 128,
  sourceCount: files.length,
  emptyCount: empty,
  patterns: [...patterns.values()]
}));

console.log("Source icons:", files.length);
console.log("Unique binary patterns:", patterns.size);
console.log("Empty patterns excluded:", empty);
console.log("Saved to:", output);
