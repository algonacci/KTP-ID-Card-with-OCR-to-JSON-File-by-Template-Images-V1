import numpy as np
import glob
import cv2
import Utils
import easyocr
import json
import os

from Params_KTP_Template import NIKPosition, detailPosition, fotoPosition, ttdPosition
from Utils_KTP_OCR import convertNIK, convertDetail

imagesPath = "Images\\"
resultImagesPath = "ResultImages\\"
resultJSONPath = "ResultJSON\\"
resultPath = "_Result_\\"

scale = 0.5
color = (0,0,255)
widthKTP = 850
heightKTP = 540
thickness = 1

reader = easyocr.Reader(['en'])

def nothing(x):
    pass

def writeJSONFile(  nik, 
                    nama, 
                    tempatL, tanggalL, bulanL, tahunL, 
                    jenisKelamin, 
                    golDarah, 
                    jalan, rt, rw, kelDesa, kecamatan, 
                    agama, statusKawin, pekerjaan, kewarganegaraan, berlakuHingga):
    
    ktp_dict = {
        "nik" : nik,
        "nama" : nama,
        "ttl" : {
            "tempatL" : tempatL,
            "tanggalL" : tanggalL,
            "bulanL" : bulanL,
            "tahunL" : tahunL
        },
        "jenisKelamin" : jenisKelamin,
        "golDarah" : golDarah,
        "alamat" : {
            "jalan" : jalan,
            "rt" : rt,
            "rw" : rw,
            "kelDesa" : kelDesa,
            "kecamatan" : kecamatan
        },
        "agama" : agama,
        "statusPerkawinan" : statusKawin,
        "pekerjaan" : pekerjaan,
        "kewarganegaraan" : kewarganegaraan,
        "berlakuHingga" : berlakuHingga
    }
    return ktp_dict

def printKTP(nik, 
             nama, 
             tempatL, tanggalL, bulanL, tahunL, 
             jenisKelamin, 
             golDarah, 
             jalan, rt, rw, kelDesa, kecamatan, 
             agama, statusKawin, pekerjaan, kewarganegaraan, berlakuHingga):
    print("-> NIK:", nik)
    print("-> Nama:", nama)
    print("-> Tempat Lahir:", tempatL)
    print("-> Tanggal Lahir:", tanggalL)
    print("-> Bulan Lahir:", bulanL)
    print("-> Tahun Lahir:", tahunL)
    print("-> Jenis Kelamin:", jenisKelamin)
    print("-> Golongan Darah:", golDarah)
    print("-> Alamat:", jalan)
    print("-> RT:", rt)
    print("-> RW:", rw)
    print("-> Kel/Desa:", kelDesa)
    print("-> Kecamatan:", kecamatan)
    print("-> Agama:", agama)
    print("-> Status Perkawinan:", statusKawin)
    print("-> Pekerjaan:", pekerjaan)
    print("-> Kewarganegaraan:", kewarganegaraan)
    print("-> Berlaku Hingga:", berlakuHingga)
    print("="*50)

def main():
    for filename in glob.glob(imagesPath+"*.jpg"):
        
        img = cv2.imread(filename)
        print(filename)

        img = Utils.resize(img, scale)
        imgEdit = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgEdit = cv2.GaussianBlur(imgEdit, (15,15), 3)
        imgEdit = cv2.adaptiveThreshold(imgEdit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,35,2)
        ##########################
        imgContours = img.copy()
        contours, hierarchy = cv2.findContours(imgEdit, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgContours, contours, -1, color, 2)
        #########################
        rectanglePoints = Utils.findRectangle(contours)
        #########################
        imgContours2 = img.copy()
        biggestContour = rectanglePoints[0]
        peri = cv2.arcLength(biggestContour, True)
        approx = cv2.approxPolyDP(biggestContour, 0.02*peri, True)
        cv2.drawContours(imgContours2, approx, -1, color, 8)
        #--------------------------------------------------
        imgFixPosition = img.copy()
        imgFixPosition, fixPosition = Utils.FixCornerPositions(imgFixPosition, approx)
        ##################################################################
        fixPosition = Utils.findPositionCorner(approx)
        pts1 = np.float32(fixPosition)
        pts2 = np.float32([[0,0],[widthKTP,0],[0,heightKTP],[widthKTP,heightKTP]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthKTP, heightKTP))
        cv2.imshow("imgWarpColored", imgWarpColored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #=================================================================
        imgRect = imgWarpColored.copy()
        NIKImage = Utils.cropRectangle(imgRect, NIKPosition)
        detailImage = Utils.cropRectangle(imgRect, detailPosition)
        fotoImage = Utils.cropRectangle(imgRect, fotoPosition)
        ttdImage = Utils.cropRectangle(imgRect, ttdPosition)
        #=================================================================
        NIKResult = reader.readtext(NIKImage)
        detailResult = reader.readtext(detailImage)
        print(detailResult)
        #=================================================================
        nik = convertNIK(NIKResult)
        nama = convertDetail(detailResult, "nama")
        tempatL, tanggalL, bulanL, tahunL = convertDetail(detailResult, "ttl")
        jenisKelamin = convertDetail(detailResult, "jenisKelamin")
        golDarah = convertDetail(detailResult, "golDarah")
        jalan, rt, rw, kelDesa, kecamatan = convertDetail(detailResult, "alamat")
        agama = convertDetail(detailResult, "agama")
        statusKawin = convertDetail(detailResult, "statusKawin")
        pekerjaan = convertDetail(detailResult, "pekerjaan")
        kewarganegaraan = convertDetail(detailResult, "kewarganegaraan")
        berlakuHingga = convertDetail(detailResult, "berlakuHingga")
        #=================================================================
        #printKTP(nik, nama, tempatL, tanggalL, bulanL, tahunL, jenisKelamin, golDarah, jalan, rt, rw, kelDesa, kecamatan, agama, statusKawin, pekerjaan, kewarganegaraan, berlakuHingga)
        #=================================================================                        
        nameFolder = nik + "_" + nama.split()[0] 
        pathSave = os.path.join(resultPath, nameFolder)
        try:
            os.makedirs(pathSave, exist_ok = True)
        except OSError as error:
            pass
        pathSave = resultPath + nameFolder + "\\"
        #===========================s======================================
        ktpDict = writeJSONFile(nik, 
                                nama, 
                                tempatL, tanggalL, bulanL, tahunL, 
                                jenisKelamin, 
                                golDarah,
                                jalan, rt, rw, kelDesa, kecamatan,
                                agama, statusKawin, pekerjaan, kewarganegaraan, berlakuHingga)

        json_object = json.dumps(ktpDict, indent = 4)
        pathJSONSave = pathSave + nameFolder + ".json"
        with open(pathJSONSave, "w") as outfile:
            outfile.write(json_object)
        #=============================================
        # ====================
        pathFotoSave = pathSave + "foto_" + nameFolder + ".jpg"
        pathTtdSave = pathSave + "ttd_" + nameFolder + ".jpg"
        cv2.imwrite(pathFotoSave, fotoImage)
        cv2.imwrite(pathTtdSave, ttdImage)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':       
    # Calling main() function 
    main()