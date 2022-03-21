# lines taken out of Poleno Operational

from zipfile import ZipFile
import sys, os, io

zip_file = sys.argv[1] + '/' + sys.argv[2]  						# reads zip file + full path from bash script

rcg_zip = ZipFile(zip_file, 'r')
json_names = [s for s in rcg_zip.namelist() if 'event.json' in s]	# list of files containing 'event.json' in the names

for i in range(len(json_names)):									# cycle through all events

    json_path_name = json_names[i]									# json name + path within zip
    json_name = os.path.basename(json_path_name)					# just json name

    with ZipFile(zip_file) as z:									# reads json contents directly from zip
        zf = z.read(json_path_name)
        json_data = json.loads(zf.decode("utf-8"))

    cam1 = json_names[i]
    cam1 = cam1[:-10] + 'rec0.png'									# 1st holo image name
    cam2 = cam1[:-8]  + 'rec1.png'									# 2nd holo image name

    img1_data = rcg_zip.read(cam1)									# reads image contents
    bytes_io1 = io.BytesIO(img1_data)								# converts binaries
    img1 = Image.open(bytes_io1)									# reads as image
    img2_data = rcg_zip.read(cam2)
    bytes_io2 = io.BytesIO(img2_data)
    img2 = Image.open(bytes_io2)

    # from that point you have the event read and ready for operations
    # I'm writing it back to zip but I believe the following should work:
    #
    #
    #		string_to_be_written = 'Some Data to be added to the ZIP file'
    #		writing_back_zip = ZipFile(zip_file, 'w')		# some options might be included e.g. compression
    #		writing_back_zip.write('from_string.txt', string_to_be_written)
    #		writing_back_zip.close()