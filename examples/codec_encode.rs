//! Encode raw packed RGB8 sRGB through zencodec. Arguments: input.rgb width height output.cool.
use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
use zencoolchic::codec::CoolChicEncoderConfig;
use zenpixels::{PixelDescriptor, PixelSlice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: codec_encode input.rgb width height output.cool".into());
    }
    let (width, height) = (args[1].parse()?, args[2].parse()?);
    let data = std::fs::read(&args[0])?;
    let pixels = PixelSlice::new(
        &data,
        width,
        height,
        width as usize * 3,
        PixelDescriptor::RGB8_SRGB,
    )?;
    let output = CoolChicEncoderConfig::default()
        .with_generic_effort(0)
        .job()
        .encoder()?
        .encode(pixels)?;
    std::fs::write(&args[3], output.data())?;
    Ok(())
}
