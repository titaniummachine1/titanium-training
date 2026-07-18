//! ACE engine CLI — true ACE reference engines (not ace-v13* / Titanium).

use ace::{
    ace_genmove, default_timeout, oracle_nodes, perft_ace_ti_timed, perft_ace_timed,
    perft_titanium_timed, run_ace_session_stdio, AceGame, AceParams, AceSearch, TimedPerftResult,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "genmove" => run_genmove(&args),
        "session" => {
            let flag = ace_engine_flag(&args).unwrap_or("ace");
            run_ace_session_stdio(flag);
        }
        "ace-bench" | "bench" => run_ace_bench(&args),
        "ace-perft" | "perft" => run_ace_perft(&args),
        "friend-perft" => ace::friend_perft::run(),
        _ => print_usage(),
    }
}

fn print_usage() {
    println!("ACE Engine 0.1.0");
    println!("  ace genmove --engine ace-v8-ti [moves...] [--time SEC] [--depth N] [--log]");
    println!("  ace session --engine ace-v8-ti          — long-lived REPL (TT persists)");
    println!("  ace ace-bench [depth] [moves...] [--cat]");
    println!("  ace ace-perft [depth] [--timeout SEC]");
    println!("  ace friend-perft [max_depth]");
    println!("  True ACE engines: ace, ace-v8, ace-v10, ace-v11, ace-cat, ace-ti,");
    println!("    ace-v8-ti, ace-v8-ti-pmc, ace-v10-ti, ace-v10-ti-pmc,");
    println!("    ace-v11-ti, ace-v11-ti-pmc, ace-pmc");
    println!("  ace-v13* and titanium-v* stay on the titanium binary.");
}

fn looks_like_algebraic_move(arg: &str) -> bool {
    let b = arg.as_bytes();
    b.len() >= 2 && b[0].is_ascii_lowercase() && b[1].is_ascii_digit()
}

fn ace_engine_flag(args: &[String]) -> Option<&str> {
    args.windows(2).find_map(|w| {
        if w[0] != "--engine" {
            return None;
        }
        match w[1].as_str() {
            "ace" | "ace-v8" | "ace-v10" | "ace-v11" | "ace-cat" | "ace-ti" | "ace-v8-ti"
            | "ace-v8-ti-pmc" | "ace-v10-ti" | "ace-v10-ti-pmc" | "ace-v11-ti"
            | "ace-v11-ti-pmc" | "ace-pmc" => Some(w[1].as_str()),
            _ => None,
        }
    })
}

fn ace_engine_mode(flag: &str) -> &'static str {
    match flag {
        "ace-cat" => "ace-cat",
        "ace-ti" | "ace-v8-ti" | "ace-v8-ti-pmc" | "ace-v10-ti" | "ace-v10-ti-pmc"
        | "ace-v11-ti" | "ace-v11-ti-pmc" => "ace-ti",
        _ => "ace",
    }
}

fn score_text(score: i32) -> String {
    const MATE: i32 = 100_000;
    const RACE_MATE: i32 = 32_000;
    let abs = score.abs();
    if abs >= MATE - 1_000 {
        let plies = MATE - abs;
        if score > 0 {
            format!("mate in {}", plies.max(0))
        } else {
            format!("mated in {}", plies.max(0))
        }
    } else if abs >= RACE_MATE - 1_000 && abs <= RACE_MATE {
        let plies = RACE_MATE - abs;
        if score > 0 {
            format!("race win in {}", plies.max(0))
        } else {
            format!("race loss in {}", plies.max(0))
        }
    } else {
        format!("cp {score}")
    }
}

fn run_genmove(args: &[String]) {
    let label = ace_engine_flag(args).unwrap_or("ace");
    let mode = ace_engine_mode(label);
    let cat = mode == "ace-cat";
    let ti_movegen = mode == "ace-ti";
    let eme0 = label.contains("pmc");
    let mut time_ms = 4000u64;
    let mut max_depth = 128i32;
    let mut full = false;
    let mut log = false;
    let mut eme = eme0;
    let mut moves = Vec::new();
    let mut i = 2usize;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--time" {
            if let Some(sec) = args.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                time_ms = (sec * 1000.0) as u64;
                i += 2;
                continue;
            }
        } else if arg == "--depth" {
            if let Some(d) = args.get(i + 1).and_then(|s| s.parse::<i32>().ok()) {
                max_depth = d;
                i += 2;
                continue;
            }
        } else if arg == "--full" {
            full = true;
            i += 1;
            continue;
        } else if arg == "--log" {
            log = true;
            i += 1;
            continue;
        } else if arg == "--eme" || arg == "--pseudo-mcts" {
            eme = true;
            i += 1;
            continue;
        } else if arg == "--engine" {
            i += 2;
            continue;
        } else if arg.starts_with("--") {
            if let Some(next) = args.get(i + 1) {
                if !next.starts_with("--") && !looks_like_algebraic_move(next) {
                    i += 2;
                    continue;
                }
            }
            i += 1;
            continue;
        } else if looks_like_algebraic_move(arg) {
            moves.push(arg.clone());
        }
        i += 1;
    }

    let params = AceParams {
        cat,
        ti_movegen,
        eme,
        time_ms,
        max_depth,
        full,
        log,
        ..Default::default()
    };
    match ace_genmove(&moves, params, label) {
        Some((algebraic, info)) => {
            if !log {
                let mut depth_json = String::new();
                for (j, e) in info.depth_log.iter().enumerate() {
                    if j > 0 {
                        depth_json.push(',');
                    }
                    let pv = e.pv.replace('\\', "\\\\").replace('"', "\\\"");
                    let score_text = score_text(e.score);
                    depth_json.push_str(&format!(
                        "{{\"depth\":{},\"score\":{},\"scoreText\":\"{}\",\"nodes\":{},\"elapsedMs\":{},\"marginalNodes\":{},\"pv\":\"{}\"}}",
                        e.depth, e.score, score_text, e.nodes, e.elapsed_ms, e.marginal_nodes, pv
                    ));
                }
                let root_score_text = score_text(info.score);
                eprintln!(
                    "info json {{\"engine\":\"{}\",\"stoppedBy\":\"{}\",\"searchDepth\":{},\"nodes\":{},\"rootScore\":{},\"rootScoreText\":\"{}\",\"whiteDist\":{},\"blackDist\":{},\"elapsedMs\":{},\"depthLog\":[{}]}}",
                    label, label, info.depth, info.nodes, info.score,
                    root_score_text,
                    info.white_dist, info.black_dist, info.ms, depth_json
                );
            }
            println!("bestmove {}", algebraic);
        }
        None => println!("bestmove (none)"),
    }
}

fn run_ace_bench(args: &[String]) {
    let use_cat = args.iter().any(|a| a == "--cat");
    let depth: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let mut g = AceGame::new();
    for arg in args.iter().skip(3) {
        if let Ok(m) = arg.parse::<i16>() {
            g.make_move(m);
        }
    }
    println!("hash {} {}", g.hash_lo, g.hash_hi);
    let mut search = if use_cat {
        AceSearch::with_cat(g)
    } else {
        AceSearch::new(g)
    };
    let r = search.think(1_000_000_000, depth, true, false, "ace-bench");
    println!(
        "{{\"move\":{},\"score\":{},\"depth\":{},\"nodes\":{},\"ms\":{}}}",
        r.mv, r.score, r.depth, r.nodes, r.ms
    );
}

fn run_ace_perft(args: &[String]) {
    use std::time::Duration;

    let depth: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let mut timeout_secs = default_timeout(depth).as_secs();
    let mut i = 3usize;
    while i < args.len() {
        if args[i] == "--timeout" {
            if let Some(sec) = args.get(i + 1).and_then(|s| s.parse::<u64>().ok()) {
                timeout_secs = sec;
                i += 2;
                continue;
            }
        }
        i += 1;
    }
    let timeout = Duration::from_secs(timeout_secs);

    fn print_line(r: &TimedPerftResult) {
        if r.timed_out {
            println!(
                "  {:12} TIMEOUT after {:.1}s (no result)",
                r.label,
                r.elapsed_ms as f64 / 1000.0
            );
            return;
        }
        let nodes = r.nodes.unwrap_or(0);
        let secs = r.elapsed_ms as f64 / 1000.0;
        let nps = if secs > 0.0 {
            nodes as f64 / secs
        } else {
            0.0
        };
        println!(
            "  {:12} nodes={} time={:.3}s nps={:.0}",
            r.label, nodes, secs, nps
        );
    }

    println!(
        "ace-perft depth={} timeout={}s (oracle perft_fast + TT vs ACE v7 wall_legal)",
        depth, timeout_secs
    );

    let ti = perft_titanium_timed(depth, timeout);
    print_line(&ti);

    let ace_ti = perft_ace_ti_timed(depth, timeout);
    print_line(&ace_ti);

    let ace = perft_ace_timed(depth, timeout);
    print_line(&ace);

    if let Some(exp) = oracle_nodes(depth) {
        println!("  oracle depth{}={}", depth, exp);
        println!(
            "  perft_fast_ok={} ace_ti_ok={} ace_native_ok={}",
            ti.nodes == Some(exp),
            ace_ti.nodes == Some(exp),
            ace.nodes == Some(exp)
        );
        if let (Some(ti_n), Some(ati_n)) = (ti.nodes, ace_ti.nodes) {
            if ti_n == ati_n {
                let ratio = ace_ti.elapsed_ms as f64 / ti.elapsed_ms.max(1) as f64;
                println!("  ace_ti vs perft_fast: {:.2}x (1.0 = same speed)", ratio);
            }
        }
        if ace.timed_out {
            println!(
                "  ace-v7-native: TIMEOUT — ported wall_legal path unusable at depth {}",
                depth
            );
        } else if let (Some(an), Some(ati_n)) = (ace.nodes, ace_ti.nodes) {
            if an == ati_n {
                let ratio = ace.elapsed_ms as f64 / ace_ti.elapsed_ms.max(1) as f64;
                println!("  ace_ti vs ace-v7-native: {:.2}x faster", ratio);
            }
        }
    }

    if ace_ti.timed_out || (ace.nodes.is_some() && ace.nodes != ace_ti.nodes) {
        std::process::exit(1);
    }
}
