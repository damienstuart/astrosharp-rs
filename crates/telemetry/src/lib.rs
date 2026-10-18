//! Tracing setup shared by the CLI and GUI: warnings and errors go to stderr, and spans
//! are exported over OTLP/HTTP (e.g. to VictoriaTraces) when an endpoint is configured.
//!
//! Settings come from the environment, falling back to `astrosharp/otel.env` in
//! `$XDG_CONFIG_HOME` (default `~/.config`, or `%APPDATA%` on Windows). The file is what
//! configures apps launched from Finder or the Start menu, which don't see shell variables.

use std::collections::HashMap;
use std::fmt::Display;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Version of AstroSharp (all crates share the workspace version).
const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Default `RUST_LOG` filter for stderr.
const DEFAULT_LOG: &str = "warn";
/// Default `ASTROSHARP_TRACE` filter for exported spans: AstroSharp's own crates only, so
/// the exporter's HTTP client and the GUI toolkit never end up in traces.
const DEFAULT_TRACE: &str = "astrosharp=info";
/// Default `OTEL_EXPORTER_OTLP_TIMEOUT`, short so an unreachable endpoint can't hold up
/// the CLI's exit for long.
const DEFAULT_TIMEOUT_MS: u64 = 5000;

/// Flushes and stops the trace exporter when dropped (or on [`Telemetry::shutdown`]).
#[must_use = "dropping Telemetry stops exporting traces"]
pub struct Telemetry {
    provider: Option<SdkTracerProvider>,
}

impl Telemetry {
    /// Sends any pending spans and stops exporting. Later calls do nothing.
    pub fn shutdown(&mut self) {
        if let Some(provider) = self.provider.take()
            && let Err(e) = provider.shutdown()
        {
            // Logged like the exporter's own errors, so RUST_LOG controls both.
            tracing::warn!("could not send traces: {e}");
        }
    }
}

impl Drop for Telemetry {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Installs the global tracing subscriber for `service_name` (`astrosharp-cli` or
/// `astrosharp-gui`). Never fails: problems are reported on stderr and leave trace
/// export off.
pub fn init(service_name: &'static str) -> Telemetry {
    let settings = Settings::load();
    let log_filter = filter(env("RUST_LOG"), DEFAULT_LOG, "RUST_LOG");
    let provider = match settings.endpoint() {
        Some(endpoint) if !settings.disabled() => match settings.provider(service_name, endpoint) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("warning: trace export is off: {e}");
                None
            }
        },
        _ => None,
    };
    let otel = provider.as_ref().map(|p| {
        let trace_filter = filter(
            settings.get("ASTROSHARP_TRACE"),
            DEFAULT_TRACE,
            "ASTROSHARP_TRACE",
        );
        tracing_opentelemetry::layer()
            .with_tracer(p.tracer("astrosharp"))
            .with_filter(trace_filter)
    });
    let stderr = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .with_filter(log_filter);
    if let Err(e) = tracing_subscriber::registry()
        .with(stderr)
        .with(otel)
        .try_init()
    {
        eprintln!("warning: could not set up tracing: {e}");
    }
    Telemetry { provider }
}

/// Marks `span` as failed, with `err` as its status message. The span must declare an
/// `otel.status_description` field (as `tracing::field::Empty`).
pub fn record_error(span: &tracing::Span, err: &dyn Display) {
    span.record("otel.status_description", err.to_string().as_str());
}

/// An [`EnvFilter`] from `directives`, or `default` if they are missing or invalid.
fn filter(directives: Option<String>, default: &str, name: &str) -> EnvFilter {
    directives
        .and_then(|d| {
            EnvFilter::try_new(&d)
                .inspect_err(|e| eprintln!("warning: ignoring {name}={d}: {e}"))
                .ok()
        })
        .unwrap_or_else(|| EnvFilter::new(default))
}

/// Settings from the environment, with the `otel.env` file as a fallback.
struct Settings {
    file: HashMap<String, String>,
}

impl Settings {
    fn load() -> Settings {
        let file = config_path()
            .and_then(|path| match std::fs::read_to_string(&path) {
                Ok(text) => Some(parse_env_file(&text)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => {
                    eprintln!("warning: could not read {}: {e}", path.display());
                    None
                }
            })
            .unwrap_or_default();
        Settings { file }
    }

    /// `key` from the environment, else from the file.
    fn get(&self, key: &str) -> Option<String> {
        env(key).or_else(|| self.file.get(key).cloned())
    }

    /// The traces endpoint. An endpoint set in the environment wins over the file's, even
    /// a generic one over the file's traces-specific one.
    fn endpoint(&self) -> Option<String> {
        traces_endpoint(env).or_else(|| traces_endpoint(|k| self.file.get(k).cloned()))
    }

    fn disabled(&self) -> bool {
        self.get("OTEL_SDK_DISABLED")
            .is_some_and(|v| v.eq_ignore_ascii_case("true"))
    }

    fn provider(
        &self,
        service_name: &'static str,
        endpoint: String,
    ) -> Result<SdkTracerProvider, String> {
        let timeout = match self.signal("TIMEOUT") {
            None => DEFAULT_TIMEOUT_MS,
            Some(ms) => ms.parse().map_err(|_| {
                format!("OTEL_EXPORTER_OTLP_TIMEOUT must be milliseconds, not {ms}")
            })?,
        };
        let headers = self
            .signal("HEADERS")
            .map(|h| parse_headers(&h))
            .unwrap_or_default();
        let exporter = SpanExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(endpoint)
            .with_timeout(Duration::from_millis(timeout))
            .with_headers(headers)
            .build()
            .map_err(|e| e.to_string())?;
        let mut resource =
            Resource::builder().with_attribute(KeyValue::new("service.version", VERSION));
        // An explicit OTEL_SERVICE_NAME is picked up by the builder's default detectors.
        if env("OTEL_SERVICE_NAME").is_none() {
            resource = resource.with_service_name(service_name);
        }
        Ok(SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(resource.build())
            .build())
    }

    /// `OTEL_EXPORTER_OTLP_TRACES_<name>`, else `OTEL_EXPORTER_OTLP_<name>`.
    fn signal(&self, name: &str) -> Option<String> {
        self.get(&format!("OTEL_EXPORTER_OTLP_TRACES_{name}"))
            .or_else(|| self.get(&format!("OTEL_EXPORTER_OTLP_{name}")))
    }
}

/// A non-empty environment variable.
fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

/// `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` as is, else `OTEL_EXPORTER_OTLP_ENDPOINT` (a base
/// URL) with `/v1/traces` appended, as the OpenTelemetry spec says.
fn traces_endpoint(get: impl Fn(&str) -> Option<String>) -> Option<String> {
    get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT").or_else(|| {
        get("OTEL_EXPORTER_OTLP_ENDPOINT")
            .map(|base| format!("{}/v1/traces", base.trim_end_matches('/')))
    })
}

fn config_path() -> Option<PathBuf> {
    let dir = env("XDG_CONFIG_HOME").map(PathBuf::from).or_else(|| {
        if cfg!(windows) {
            env("APPDATA").map(PathBuf::from)
        } else {
            env("HOME").map(|home| PathBuf::from(home).join(".config"))
        }
    })?;
    Some(dir.join("astrosharp").join("otel.env"))
}

/// `KEY=value` lines; blank lines and `#` comments are skipped, and an `export ` prefix
/// and quotes around the value are allowed.
fn parse_env_file(text: &str) -> HashMap<String, String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            let (key, value) = line
                .strip_prefix("export ")
                .unwrap_or(line)
                .split_once('=')?;
            let value = value.trim();
            let unquoted = ['"', '\'']
                .into_iter()
                .find_map(|q| value.strip_prefix(q)?.strip_suffix(q))
                .unwrap_or(value);
            Some((key.trim().to_string(), unquoted.to_string()))
        })
        .collect()
}

/// `key1=value1,key2=value2`.
fn parse_headers(s: &str) -> HashMap<String, String> {
    s.split(',')
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            Some((k.trim().to_string(), v.trim().to_string()))
        })
        .filter(|(k, _)| !k.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_file_syntax() {
        let vars = parse_env_file(
            "# VictoriaTraces\n\
             OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:10428/insert/opentelemetry\n\
             \n\
             export ASTROSHARP_TRACE = \"astrosharp=debug\"\n\
             OTEL_EXPORTER_OTLP_HEADERS='VT-Extra-Fields=host=m5'\n\
             not a setting\n",
        );
        assert_eq!(vars.len(), 3);
        assert_eq!(
            vars["OTEL_EXPORTER_OTLP_ENDPOINT"],
            "http://localhost:10428/insert/opentelemetry"
        );
        assert_eq!(vars["ASTROSHARP_TRACE"], "astrosharp=debug");
        assert_eq!(
            vars["OTEL_EXPORTER_OTLP_HEADERS"],
            "VT-Extra-Fields=host=m5"
        );
    }

    #[test]
    fn endpoint_resolution() {
        let vars = |pairs: &[(&str, &str)]| {
            let map: HashMap<String, String> = pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            traces_endpoint(|k| map.get(k).cloned())
        };
        assert_eq!(vars(&[]), None);
        assert_eq!(
            vars(&[(
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "http://h:10428/insert/opentelemetry/"
            )]),
            Some("http://h:10428/insert/opentelemetry/v1/traces".into())
        );
        assert_eq!(
            vars(&[
                ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://a"),
                ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://b/traces"),
            ]),
            Some("http://b/traces".into())
        );
    }

    #[test]
    fn headers() {
        let h = parse_headers("VT-Extra-Fields=host=m5, Authorization=Bearer x,junk");
        assert_eq!(h.len(), 2);
        assert_eq!(h["VT-Extra-Fields"], "host=m5");
        assert_eq!(h["Authorization"], "Bearer x");
    }
}
