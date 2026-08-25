// Stamp the build with the commit it came from. Deploys go through `git pull` + `cargo build`
// + copy, and when any step silently no-ops the running engine is an old binary that looks
// identical from the outside -- which has now cost several rounds of debugging a fix that was
// never actually deployed.
fn main() {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=LLMFAST_COMMIT={commit}");
    // Rebuild when HEAD moves, so the stamp cannot go stale on an otherwise-cached build.
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/refs/heads/main");
}
