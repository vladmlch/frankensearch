use std::cell::RefCell;
use std::io::Write;
use std::path::PathBuf;

struct Record {
    label: String,
    shape: Vec<usize>,
    dtype: &'static str,
    bytes: Vec<u8>,
}

struct Recorder {
    directory: PathBuf,
    records: Vec<Record>,
    bytes: usize,
    detail: bool,
}

thread_local! {
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

pub struct Guard;

pub fn begin() -> Option<Guard> {
    let directory = PathBuf::from(std::env::var_os("FSFS_NATIVE_CERTIFICATE_TRACE_DIR")?);
    // create_dir refuses an existing directory, including a symlink.
    std::fs::create_dir(&directory).expect("create new certificate trace directory");
    RECORDER.with(|slot| {
        let mut slot = slot.borrow_mut();
        assert!(slot.is_none(), "certificate trace already active");
        *slot = Some(Recorder {
            directory,
            records: Vec::new(),
            bytes: 0,
            detail: false,
        });
    });
    Some(Guard)
}

pub fn active() -> bool {
    RECORDER.with(|slot| slot.borrow().is_some())
}

pub fn detail(enabled: bool) {
    RECORDER.with(|slot| {
        if let Some(recorder) = slot.borrow_mut().as_mut() {
            recorder.detail = enabled;
        }
    });
}

fn record(label: &str, shape: &[usize], dtype: &'static str, bytes: Vec<u8>) {
    RECORDER.with(|slot| {
        if let Some(recorder) = slot.borrow_mut().as_mut() {
            recorder.bytes += bytes.len();
            assert!(
                recorder.bytes <= 32 * 1024 * 1024,
                "certificate trace exceeded 32 MiB"
            );
            recorder.records.push(Record {
                label: label.to_owned(),
                shape: shape.to_vec(),
                dtype,
                bytes,
            });
        }
    });
}

pub fn floats(label: &str, shape: &[usize], values: &[f32]) {
    if active() {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        record(
            label,
            shape,
            "f32-le",
            values
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        );
    }
}

pub fn detailed(label: &str, shape: &[usize], values: &[f32]) {
    if RECORDER.with(|slot| slot.borrow().as_ref().is_some_and(|r| r.detail)) {
        floats(label, shape, values);
    }
}

pub fn ids(label: &str, values: &[i64]) {
    if active() {
        record(
            label,
            &[values.len()],
            "i64-le",
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        );
    }
}

impl Guard {
    pub fn flush(self) {
        let recorder = RECORDER
            .with(|slot| slot.borrow_mut().take())
            .expect("active trace");
        let mut metadata = Vec::new();
        for (ordinal, record) in recorder.records.into_iter().enumerate() {
            let filename = format!("{ordinal:04}.bin");
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(recorder.directory.join(&filename))
                .expect("create trace record");
            file.write_all(&record.bytes).expect("write trace record");
            metadata.push(serde_json::json!({
                "ordinal": ordinal, "label": record.label, "shape": record.shape,
                "dtype": record.dtype, "file": filename,
            }));
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(recorder.directory.join("manifest.json"))
            .expect("create trace manifest");
        serde_json::to_writer_pretty(&mut file, &metadata).expect("write trace manifest");
        drop(self);
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        RECORDER.with(|slot| {
            let _ = slot.borrow_mut().take();
        });
    }
}
