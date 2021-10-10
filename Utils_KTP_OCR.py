

def convertNIK(NIK):
    NIK = NIK[0][1]
    return NIK

def convertDetail(detail, input):

    ### Solution to evercome problem "ttl" (Tempat, tgl-bln-thn) and (Tempat) - (tgl-bln-thn)
    if len(detail) == 13:
        i = 0
    else:
        i = 1

    if input == "ttl":
        ### Solution to evercome problem "ttl" (Tempat, tgl-bln-thn) and (Tempat) - (tgl-bln-thn)
        
        if len(detail) == 13:
            detail = detail[1][1]
            tempatL = detail[:-11]

            waktuL = detail[-10:]
            tanggalL = int(waktuL[-10:-8])
            bulanL = int(waktuL[-7:-5])
            tahunL = int(waktuL[-4:])
        else:
            tempatL = detail[1][1]

            waktuL = detail[2][1]
            tanggalL = int(waktuL[-10:-8])
            bulanL = int(waktuL[-7:-5])
            tahunL = int(waktuL[-4:])
        return tempatL, tanggalL, bulanL, tahunL

    elif input == "alamat":
        jalan = detail[4+i][1]
        rtrw = detail[5+i][1]
        rt = rtrw[:3]
        rw = rtrw[-3:]
        kelDesa = detail[6+i][1]
        kecamatan = detail[7+i][1]
        return jalan, rt, rw, kelDesa, kecamatan

    else:
        if input == "nama":
            detail = detail[0][1]
        
        elif input == "jenisKelamin":
            detail = detail[2+i][1]
            if detail[:3] == "LAK":
                detail = "LAKI-LAKI"
            elif detail[:2] == "PE":
                detail = "PEREMPUAN"
            
        elif input == "golDarah":
            detail = detail[3+i][1]
            if detail[-1:] == "0" or detail[-1:] == "O":
                detail = "O"
            elif detail[-1:] == "A":
                detail = "A"
            elif detail[-2:] == "AB":
                detail = "AB"
            elif detail[-1:] == "B":
                detail = "B"

        elif input == "agama":
            detail = detail[8+i][1]
            if detail[:2] == "IS":
                detail = "ISLAM"
            elif detail[:2] == "BU":
                detail = "BUDDHA"
            elif detail[:2] == "HI":
                detail = "HINDU"
            elif detail[:2] == "KO":
                detail = "KONGHUCU"
            elif detail[8:9] == "P":
                detail = "KRISTEN PROTESTAN"
            elif detail[8:10] == "KA":
                detail = "KRISTEN KATOLIK"

        elif input == "statusKawin":
            detail = detail[9+i][1]
            if detail[:2] == "BE":
                detail = "BELUM KAWIN"
            elif detail[:2] == "KA":
                detail = "KAWIN"
            elif detail[-1:] == "P":
                detail = "CERAI HIDUP"
            elif detail[-1:] ==  "I":
                detail = "CERAI MATI"

        elif input == "pekerjaan":
            detail = detail[10+i][1]
            
        elif input == "kewarganegaraan":
            detail = detail[11+i][1]
            
        elif input == "berlakuHingga": 
            detail = detail[12+i][1]   
            if detail[:1] == "S":
                detail = "SEUMUR HIDUP"        

        return detail