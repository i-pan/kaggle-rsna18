echo "Extracting metadata from DICOM Files ..."
python src/etl/1_GetMetadata.py
echo "Creating stratified 10-fold cross-validation ..."
python src/etl/2_AssignCVFolds.py
echo "Converting DICOMs to PNGs ..." 
python src/etl/3_ConvertDICOM2PNG.py 
echo "Creating resized numpy arrays from PNGs ..."
python src/etl/4_CreateResizedNumpyArrays.py
echo "Creating flipped test images for later inference ..."
python src/etl/5_FlipTestImages.py
echo "Transforming data into COCO format ..."
python src/etl/6_COCOify.py train
python src/etl/6_COCOify.py test
echo "Generating concatenated images for ConcatRetinaNet ..."
python src/etl/7_CreateConcatImages.py 
echo "Generating annotations for training keras-retinanet ..."
python src/etl/8_GenerateConcatRetinaAnnotations.py 
python src/etl/9_GeneratePositiveRetinaAnnotations.py
echo "DONE !" 
