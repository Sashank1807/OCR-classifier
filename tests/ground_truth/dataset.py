"""
Verified Ground Truth reference dataset for Stock & Sales Statement Document Benchmarking.
"""

GROUND_TRUTH_DATA = {
    "01_saraswati_drug_agency.pdf": {
        "doc_type": "PDF (Vector/Scanned Multi-Page)",
        "distributor_name": "SARASWATI DRUG AGENCY",
        "period": "01/04/2026 Upto 30/04/2026",
        "company": "HETRO HEALTHCARE KRIS",
        "total_rows": 42,
        "summary_totals": {
            "opening_bal_value": 219132.71,
            "receipt_value": 130388.90,
            "sales_value": 10960.75,
            "closing_bal_value": 339301.80,
            "near_expiry": 63873
        },
        "sample_numeric_values": [
            184.80, 458.70, 646.80, 523.19, 16607.04, 16355.15, 9591.12, 
            17597.84, 377.14, 17152.33, 692.06, 6040.42, 802.89, 40.74, 
            1601.78, 687.58, 353.16, 1928.94, 2414.48, 17113.32, 6338.08, 
            11123.66, 5561.89, 2263.67, 3422.66, 26311.46, 30400.26, 1628.70, 
            53810.54, 4629.56, 1177.85, 22303.64, 85148.57, 110063.65, 39315.73, 
            13512.10, 51828.00, 4177.97, 533.61, 1551.48, 189.84, 272.79, 
            181.69, 182.96, 2756.29, 3987.33, 2996.91, 2737.00, 3169.53, 
            3699.68, 2327.77, 1867.57
        ]
    },
    "02_rs_pharmaceuticals_crumpled.jpg": {
        "doc_type": "Photographed Crumpled Paper Document",
        "distributor_name": "R.S. PHARMACEUTICALS",
        "gstin": "09ESEFS1996R1Z1",
        "division": "HETERO HEALION",
        "period": "01-04-2026 - 30-04-2026",
        "total_rows": 29,
        "summary_totals": {
            "opening_stock_qty": 2036,
            "purchases_qty": 1453,
            "total_stock_qty": 3489,
            "sales_qty": 1683,
            "closing_stock_qty": 1806,
            "opening_value": 103686,
            "purchases_value": 92489,
            "total_value": 196171,
            "sales_value": 101960,
            "closing_value": 113196
        },
        "sample_numeric_values": [
            75, 150, 225, 37.73, 600, 590, 10, 69.76, 140, 64.3, 
            130, 44.44, 97, 26, 71, 73.47, 90, 15, 75, 42.86, 
            75, 30, 45, 37.45, 381, 73.47, 60, 140, -80, 83.19, 
            75, 140, 215, 91.84, 130, 65.15, 77, 26, 51, 78.12, 
            140, 65, 75, 41.19, 23, 47.58, 17, 42, 59, 60.98, 
            42, 66.18, 3, 42, 45, 50.78, 28, 91.35, -3, 57.41, 
            39, 133.1, 25, 27.93, 1, 300, 301, 220, 81, 53.71, 
            570, 569, 1, 17.34
        ]
    },
    "03_ravi_krishna_excel_screen.jpg": {
        "doc_type": "Photograph of Phone Screen (Excel Spreadsheet Grid)",
        "distributor_name": "M/S RAVI KRISHNA AGENCIES",
        "address": "# 31-35-39, VIVEKANANDA COLONY, ASSAM GARDENS, VISAKHAPATNAM-530004",
        "company": "HETRO HEALION DIV",
        "period": "01-Apr-26 To: 30-Apr-26",
        "total_rows": 2,
        "summary_totals": {
            "opening_value": 20393.6,
            "purchase_value": 23781.0,
            "sales_value": 7122.15,
            "closing_value": 43604.6
        },
        "sample_numeric_values": [
            178, 120, 298, 298, 1, 370, 371, 180, 191, 20393.6, 23781.0, 7122.15, 43604.6
        ]
    },
    "04_sri_venkateswara_erp_screen.jpg": {
        "doc_type": "Photograph of Desktop Screen (ERP Software UI)",
        "distributor_name": "SRI VENKATESWARA MEDICAL AGENCIES",
        "company": "HETERO HEALTHCARE",
        "period": "01/04/2026 To 29/04/2026",
        "total_rows": 8,
        "summary_totals": {
            "opening_value": 0.00,
            "purchase_value": 37672.50,
            "sales_value": 29338.76,
            "closing_value": 14240.58
        },
        "sample_numeric_values": [
            1, 1, 94.50, 232, 42, 91, 5, 36, 78, 60, 130, 78, 52, 4115.59, 
            22, 124, 195, 91, 138, 8980.49, 7, 12, 26, 22, 50, 1050.00, 
            37672.50, 29338.76, 14240.58
        ]
    },
    "05_maruthi_agencies_cracked_screen.jpg": {
        "doc_type": "Photograph of Phone Screen with Cracked Glass & Reflections",
        "distributor_name": "MARUTHI AGENCIES",
        "address": "3-4-550,GR FLR(ROOM NO.1 &2),BACKSIDE PORTION OF 1ST FLR 3RD FLR(COMPLETE),PENT HOUSE NO.1&2,NARAYANAGUDA HYDERABAD",
        "company": "HETERO KRIS",
        "period": "01-Apr-2026 To 30-Apr-2026",
        "total_rows": 25,
        "summary_totals": {
            "opening_value": 73742.95,
            "purchase_value": 5896.40,
            "sales_value": 26921.06,
            "closing_value": 52718.30
        },
        "near_expiry_batches": [
            {"product": "C-FURO CV 625MG TAB", "batch": "HH2506333", "expiry": "11/2026", "qty": 9.0},
            {"product": "C-FURO CV DRY SYP", "batch": "VBD250057", "expiry": "10/2026", "qty": 1.0},
            {"product": "C-FURO CV DRY SYP", "batch": "VBD250061", "expiry": "11/2026", "qty": 4.0},
            {"product": "C-FURO DRY SYP", "batch": "CD250161", "expiry": "10/2026", "qty": 49.0},
            {"product": "C-FURO DRY SYP", "batch": "CD250134", "expiry": "9/2026", "qty": 1.0},
            {"product": "OFFICE 50 DRY SYP", "batch": "DS25008", "expiry": "8/2026", "qty": 1.0},
            {"product": "RABEZ 20MG TAB", "batch": "2GT25490B", "expiry": "12/2026", "qty": 69.0},
            {"product": "RABEZ D CAP", "batch": "2GC25493A", "expiry": "12/2026", "qty": 31.0},
            {"product": "RIFGUARD 400MG TAB", "batch": "2GT24785A", "expiry": "9/2026", "qty": 6.0}
        ],
        "sample_numeric_values": [
            10, 10, 91, 6, 6, 68, 15, 15, 68, 12, 5, 7, 222, 56, 56, 306, 
            260, 145, 115, 80, 5, 3, 2, 336, 11, 11, 190, 12, 20, 32, 32, 
            19, 15, 4, 244, 68, 30, 38, 214, 11, 6, 5, 336, 19, 8, 11, 91, 
            9, 20, 29, 8, 21, 91, 76, 7, 69, 231, 301, 17, 284, 190, 46, 4, 
            42, 153, 14, 4, 10, 487, 38, 15, 23, 306, 73742.95, 5896.40, 
            26921.06, 52718.30
        ]
    }
}
