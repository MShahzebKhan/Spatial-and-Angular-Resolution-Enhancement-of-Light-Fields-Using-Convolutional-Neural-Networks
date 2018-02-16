function LF = LFXReadStanfordIllum( FName, LensletSize )

LensletSize = LFDefaultVal('LensletSize',[14,14]);

Img = imread( FName );
ImgSize = size(Img(:,:,1));

LFSize(3:4) = ceil(ImgSize./LensletSize);
PadSize = LFSize(3:4) .* LensletSize;
PadAmt = PadSize-ImgSize;

LFSize(1:2) = PadSize ./ LFSize(3:4);
LFSize(5) = 3;

ImgPad = padarray(Img, PadAmt, 0,'post');

LF = reshape(ImgPad, LFSize([1,3,2,4,5]));
LF = permute(LF, [1,3,2,4,5]);

end