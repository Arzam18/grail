use wide::{f32x16, i16x32};

pub type SimdF32 = f32x16;
pub type SimdI16 = i16x32;

pub const SIMD_WIDTH_F32: usize = 16;
pub const SIMD_WIDTH_I16: usize = 32;

// Different architectures can keep a different number of SIMD vectors in registers at once.
// Needed because stuff like dot products can then load the large input vector once per batch instead of once per row (which is expensive).
// Tuned (ish).
// Future me: This was actually not the case, and it is CPU specific. So just hardcode 4 for now since it is faster...
// TODO: Look more into this.
pub const SIMD_BATCH_SIZE: usize = 4;

pub fn simd_relu(values: &mut [f32]) {
    let len = values.len();
    let mut i = 0;
    let zeros = SimdF32::splat(0.0);

    while i + SIMD_WIDTH_F32 <= len {
        let chunk = SimdF32::new(values[i..i + SIMD_WIDTH_F32].try_into().unwrap());
        values[i..i + SIMD_WIDTH_F32].copy_from_slice(chunk.max(zeros).as_array());
        i += SIMD_WIDTH_F32;
    }

    for val in &mut values[i..len] {
        *val = val.max(0.0);
    }
}

pub fn simd_add(dest: &mut [f32], src: &[f32]) {
    let len = dest.len();
    let mut i = 0;

    while i + SIMD_WIDTH_F32 <= len {
        let d = SimdF32::new(dest[i..i + SIMD_WIDTH_F32].try_into().unwrap());
        let s = SimdF32::new(src[i..i + SIMD_WIDTH_F32].try_into().unwrap());
        dest[i..i + SIMD_WIDTH_F32].copy_from_slice((d + s).as_array());
        i += SIMD_WIDTH_F32;
    }

    for j in i..len {
        dest[j] += src[j];
    }
}

pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();

    let mut sum_vec = SimdF32::splat(0.0);
    let mut i = 0;

    while i + SIMD_WIDTH_F32 <= len {
        let a_vec = SimdF32::new(a[i..i + SIMD_WIDTH_F32].try_into().unwrap());
        let b_vec = SimdF32::new(b[i..i + SIMD_WIDTH_F32].try_into().unwrap());
        sum_vec += a_vec * b_vec;
        i += SIMD_WIDTH_F32;
    }

    let mut sum = sum_vec.reduce_add();
    for j in i..len {
        sum += a[j] * b[j];
    }

    sum
}

pub fn simd_dot_product_batched<const N: usize>(input: &[f32], rows: [&[f32]; N]) -> [f32; N] {
    let len = input.len();
    let (input, tail) = input.as_chunks::<SIMD_WIDTH_F32>();
    let rows = rows.map(|row| row[..len].as_chunks::<SIMD_WIDTH_F32>());
    let mut sums = [SimdF32::splat(0.0); N];

    for (i, chunk) in input.iter().enumerate() {
        let x = SimdF32::new(*chunk);
        for (sum, (weights, _)) in sums.iter_mut().zip(&rows) {
            *sum += x * SimdF32::new(weights[i]);
        }
    }

    let mut dots = sums.map(|sum| sum.reduce_add());
    for (dot, (_, weights)) in dots.iter_mut().zip(&rows) {
        for j in 0..tail.len() {
            *dot += tail[j] * weights[j];
        }
    }
    dots
}
