#ifndef DATABASE_H
#define DATABASE_H

#include <iostream>
#include <pqxx/pqxx>
#include "compute.h"
#include "utils.h"

void saveColorHistograms(int id, std::vector<cv::Mat> histograms);
void saveLBPFeatures(int id, cv::Mat lbpFeatures);
void saveHOGFeatures(int id, cv::Mat hogFeatures);

//Update with winhost ip
std::string connString = "host=172.24.96.1 port=5432 dbname=shoes user=postgres password=root";
pqxx::connection conn(connString.c_str());

void saveShoeProperties(int id, std::vector<cv::Mat> RGBHistograms, cv::Mat lbpHistogram, cv::Mat hogDescriptor) {
    try {
        if (!conn.is_open()) {
            std::cerr << "Database connection not open" << std::endl;
            return;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return;
    }

    try {
        saveColorHistograms(id, RGBHistograms);
        saveLBPFeatures(id, lbpHistogram);
        saveHOGFeatures(id, hogDescriptor);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return;
    }
}



void saveColorHistograms(int shoeImageId, std::vector<cv::Mat> histograms) {
    try {
        pqxx::work txn(conn);

        pqxx::binarystring redHistBinary(reinterpret_cast<const std::byte*>(histograms[0].data), histograms[0].total() * histograms[0].elemSize());
        pqxx::binarystring greenHistBinary(reinterpret_cast<const std::byte*>(histograms[1].data), histograms[1].total() * histograms[1].elemSize());
        pqxx::binarystring blueHistBinary(reinterpret_cast<const std::byte*>(histograms[2].data), histograms[2].total() * histograms[2].elemSize());

        txn.exec_params("INSERT INTO public.evaluate_shoehistograms (shoe_image_id, red_histogram, green_histogram, blue_histogram) VALUES ($1, $2, $3, $4)",
            shoeImageId,
            redHistBinary,
            greenHistBinary,
            blueHistBinary
        );
        txn.commit();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

std::vector<cv::Mat> getRGBHistogramsByShoeImageId(int shoeImageId) {
    std::vector<cv::Mat> histograms;
    pqxx::work txn(conn);
    try {
        if (!conn.is_open()) {
            std::cerr << "Can't open database" << std::endl;
            return histograms;
        }

        pqxx::result res = txn.exec_params(
            R"(
                SELECT red_histogram, green_histogram, blue_histogram
                FROM public.evaluate_shoehistograms
                WHERE shoe_image_id = $1;
            )",
            shoeImageId
        );

        for (const auto& row: res) {
            pqxx::binarystring redHistBinary = row["red_histogram"].as<pqxx::binarystring>();
            pqxx::binarystring greenHistBinary = row["green_histogram"].as<pqxx::binarystring>();
            pqxx::binarystring blueHistBinary = row["blue_histogram"].as<pqxx::binarystring>();

            cv::Mat redHist = cv::Mat(
                redHistBinary.size() / sizeof(float),
                1,
                CV_32F,
                (void*)redHistBinary.data()
            );
            cv::Mat greenHist = cv::Mat(
                greenHistBinary.size() / sizeof(float),
                1,
                CV_32F,
                (void*)greenHistBinary.data()
            );
            cv::Mat blueHist = cv::Mat(
                blueHistBinary.size() / sizeof(float),
                1,
                CV_32F,
                (void*)blueHistBinary.data()
            );

            histograms.push_back(redHist.clone());
            histograms.push_back(greenHist.clone());
            histograms.push_back(blueHist.clone());
        }


    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    return histograms;
}

std::vector<std::vector<cv::Mat>> getRGBHistograms() {
    std::vector<std::vector<cv::Mat>> histograms;
    pqxx::work txn(conn);
    try {
        if (!conn.is_open()) {
            std::cerr << "Can't open database" << std::endl;
            return histograms;
        }

        pqxx::result res = txn.exec_params(
            R"(
                SELECT red_histogram, green_histogram, blue_histogram
                FROM public.evaluate_shoehistograms;
            )"
        );
        for (const auto& row: res) {
            pqxx::binarystring redHistBinary = row["red_histogram"].as<pqxx::binarystring>();
            pqxx::binarystring greenHistBinary = row["green_histogram"].as<pqxx::binarystring>();
            pqxx::binarystring blueHistBinary = row["blue_histogram"].as<pqxx::binarystring>();

            cv::Mat redHist = cv::Mat(
                redHistBinary.size() / sizeof(float),
                1,
                CV_32F,
                (void*)redHistBinary.data()
            );
            cv::Mat greenHist = cv::Mat(
                greenHistBinary.size() / sizeof(float),
                1,
                CV_32F,
                (void*)greenHistBinary.data()
            );
            cv::Mat blueHist = cv::Mat(
                blueHistBinary.size() / sizeof(float),
                1,
                CV_32F,
                (void*)blueHistBinary.data()
            );

            std::vector<cv::Mat> shoeHistograms;
            shoeHistograms.push_back(redHist.clone());
            shoeHistograms.push_back(greenHist.clone());
            shoeHistograms.push_back(blueHist.clone());

            histograms.push_back(shoeHistograms);
        }


    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    return histograms;
}



void saveLBPFeatures(int shoeImageId, cv::Mat lbpFeatures) {
    try {
        pqxx::work txn(conn);

        std::cout << "Saving LBP features to database" << std::endl;
        std::cout << "lbpFeatures size: " << lbpFeatures.size() << std::endl;
        std::cout << "lbpFeatures type: " << lbpFeatures.type() << std::endl;

        pqxx::binarystring lbpBinary(reinterpret_cast<std::byte*>(lbpFeatures.data), lbpFeatures.total() * lbpFeatures.elemSize());
        int lbpRows = lbpFeatures.rows;
        int lbpCols = lbpFeatures.cols;

        txn.exec_params(
            "INSERT INTO public.evaluate_shoelbp (lbp_histogram, lbp_rows, lbp_columns, shoe_image_id) VALUES ($1, $2, $3, $4)",
            lbpBinary,
            lbpRows,
            lbpCols,
            shoeImageId
        );

        txn.commit();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

cv::Mat getLBPFeaturesByShoeImageId(int shoeImageId) {
    cv::Mat lbpFeatures;
    pqxx::work txn(conn);
    try {
        if (!conn.is_open()) {
            std::cerr << "Can't open database" << std::endl;
            return lbpFeatures;
        }

        pqxx::result res = txn.exec_params(
            R"(
                SELECT lbp_histogram, lbp_rows, lbp_columns
                FROM public.evaluate_shoelbp
                WHERE shoe_image_id = $1;
            )",
            shoeImageId
        );

        for (const auto& row : res) {
            std::cout << "LBP histogram found" << std::endl;
            std::cout << "LBP histogram size: " << row["lbp_histogram"].size() << std::endl;
            pqxx::binarystring lbpBinary = row["lbp_histogram"].as<pqxx::binarystring>();
            std::cout << "LBP histogram binary size: " << lbpBinary.size() << std::endl;
            lbpFeatures = cv::Mat(
                row["lbp_rows"].as<int>(),
                row["lbp_columns"].as<int>(),
                CV_32F,
                (void*)lbpBinary.data()
            );
            std::cout << "LBP histogram size: " << lbpFeatures.size() << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    return lbpFeatures.clone();
}

std::vector<cv::Mat> getLBPHistograms() {
    std::vector<cv::Mat> lbpHistograms;
    pqxx::work txn(conn);
    try {
        if (!conn.is_open()) {
            std::cerr << "Can't open database" << std::endl;
            return lbpHistograms;
        }

        pqxx::result res = txn.exec_params(
            R"(
                SELECT lbp_histogram, lbp_rows, lbp_columns
                FROM public.evaluate_shoelbp;
            )"
        );

        for (const auto& row : res) {
            pqxx::binarystring lbpBinary = row["lbp_histogram"].as<pqxx::binarystring>();
            cv::Mat lbpHistogram = cv::Mat(
                row["lbp_rows"].as<int>(),
                row["lbp_columns"].as<int>(),
                CV_32F,
                (void*)lbpBinary.data()
            );
            lbpHistograms.push_back(lbpHistogram.clone());
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    return lbpHistograms;
}



void saveHOGFeatures(int shoeImageId, cv::Mat hogFeatures) {
    try {
        pqxx::work txn(conn);

        std::cout << "Saving HOG features to database" << std::endl;
        std::cout << "hogFeatures size: " << hogFeatures.size() << std::endl;
        std::cout << "hogFeatures type: " << hogFeatures.type() << std::endl;

        pqxx::binarystring hogBinary(reinterpret_cast<std::byte*>(hogFeatures.data), hogFeatures.total() * hogFeatures.elemSize());
        int hogRows = hogFeatures.rows;
        int hogCols = hogFeatures.cols;

        txn.exec_params(
            "INSERT INTO public.evaluate_shoehog (hog_descriptor, hog_rows, hog_columns, shoe_image_id) VALUES ($1, $2, $3, $4)",
            hogBinary,
            hogRows,
            hogCols,
            shoeImageId
        );
        txn.commit();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

cv::Mat getHOGFeaturesByShoeImageId(int shoe_image_id) {
    cv::Mat hogFeatures;
    pqxx::work txn(conn);
    try {
        if (!conn.is_open()) {
            std::cerr << "Can't open database" << std::endl;
            return hogFeatures;
        }

        pqxx::result res = txn.exec_params(
            R"(
                SELECT hog_descriptor, hog_rows, hog_columns
                FROM public.evaluate_shoehog
                WHERE shoe_image_id = $1;
            )",
            shoe_image_id
        );

        for (const auto& row : res) {
            std::cout << "HOG descriptor found" << std::endl;
            std::cout << "HOG descriptor size: " << row["hog_descriptor"].size() << std::endl;
            pqxx::binarystring hogBinary = row["hog_descriptor"].as<pqxx::binarystring>();
            std::cout << "HOG descriptor binary size: " << hogBinary.size() << std::endl;

            hogFeatures = cv::Mat(
                row["hog_rows"].as<int>(),
                row["hog_columns"].as<int>(),
                CV_32F,
                (void*)hogBinary.data()
            );
            std::cout << "HOG descriptor size: " << hogFeatures.size() << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    return hogFeatures.clone();
}

std::vector<cv::Mat> getHOGFeatures() {
    std::vector<cv::Mat> hogFeatures;
    pqxx::work txn(conn);
    try {
        if (!conn.is_open()) {
            std::cerr << "Can't open database" << std::endl;
            return hogFeatures;
        }

        pqxx::result res = txn.exec_params(
            R"(
                SELECT hog_descriptor, hog_rows, hog_columns
                FROM public.evaluate_shoehog;
            )"
        );

        for (const auto& row : res) {
            pqxx::binarystring hogBinary = row["hog_descriptor"].as<pqxx::binarystring>();
            cv::Mat hogDescriptor = cv::Mat(
                row["hog_rows"].as<int>(),
                row["hog_columns"].as<int>(),
                CV_32F,
                (void*)hogBinary.data()
            );
            hogFeatures.push_back(hogDescriptor.clone());
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    return hogFeatures;

}


// Get all properties of all shoe images
// Join tables and fetch them together in a single big object
struct ShoePropertiesList {
    std::vector<int> shoeImageIds;
    std::vector<std::vector<cv::Mat>> RGBHistograms;
    std::vector<cv::Mat> LBPHistograms;
    std::vector<cv::Mat> HOGFeatures;
};

ShoePropertiesList getShoeProperties() {
    ShoePropertiesList shoePropertiesList;
    pqxx::work txn(conn);
    try {
        if (!conn.is_open()) {
            std::cerr << "Can't open database" << std::endl;
            return shoePropertiesList;
        }

        pqxx::result res = txn.exec(
            R"(
                SELECT hist.shoe_image_id, red_histogram, green_histogram, blue_histogram,
                    lbp_histogram, lbp_rows, lbp_columns,
                    hog_descriptor, hog_rows, hog_columns
                FROM public.evaluate_shoehistograms as hist
                JOIN public.evaluate_shoelbp as lbp ON hist.shoe_image_id = lbp.shoe_image_id
                JOIN public.evaluate_shoehog as hog ON hist.shoe_image_id = hog.shoe_image_id;
            )"
        );

        for (const auto& row : res) {
            int shoeImageId = row["shoe_image_id"].as<int>();

            pqxx::binarystring redHistBinary = row["red_histogram"].as<pqxx::binarystring>();
            pqxx::binarystring greenHistBinary = row["green_histogram"].as<pqxx::binarystring>();
            pqxx::binarystring blueHistBinary = row["blue_histogram"].as<pqxx::binarystring>();

            pqxx::binarystring lbpBinary = row["lbp_histogram"].as<pqxx::binarystring>();
            pqxx::binarystring hogBinary = row["hog_descriptor"].as<pqxx::binarystring>();

            //test print all extracted data
            // std::cout << "hist shoeimageid: " << shoeImageId << std::endl;
            // std::cout << "redHistBinary size: " << redHistBinary.size() << std::endl;
            // std::cout << "greenHistBinary size: " << greenHistBinary.size() << std::endl;
            // std::cout << "blueHistBinary size: " << blueHistBinary.size() << std::endl;
            // std::cout << "lbpBinary size: " << lbpBinary.size() << std::endl;
            // std::cout << "hogBinary size: " << hogBinary.size() << std::endl;
            // std::cout << std::endl;

            // std::cout << "before converting mat" << std::endl;
            // cv::Mat redHist = binarystringToMat(redHistBinary);
            // cv::Mat greenHist = binarystringToMat(greenHistBinary);
            // cv::Mat blueHist = binarystringToMat(blueHistBinary);

            // std::cout << "before converting lbp and hog" << std::endl;
            // cv::Mat lbpFeatures = binarystringToMat(lbpBinary, row["lbp_rows"].as<int>(), row["lbp_columns"].as<int>());
            // cv::Mat hogFeatures = binarystringToMat(hogBinary, row["hog_rows"].as<int>(), row["hog_columns"].as<int>());

            cv::Mat redHist = cv::Mat(
                redHistBinary.size() / sizeof(float),
                1,
                CV_32F,
                (void*)redHistBinary.data()
            );
            cv::Mat greenHist = cv::Mat(
                greenHistBinary.size() / sizeof(float),
                1,
                CV_32F,
                (void*)greenHistBinary.data()
            );
            cv::Mat blueHist = cv::Mat(
                blueHistBinary.size() / sizeof(float),
                1,
                CV_32F,
                (void*)blueHistBinary.data()
            );

            cv::Mat lbpFeatures = cv::Mat(
                row["lbp_rows"].as<int>(),
                row["lbp_columns"].as<int>(),
                CV_32F,
                (void*)lbpBinary.data()
            );
            cv::Mat hogFeatures = cv::Mat(
                row["hog_rows"].as<int>(),
                row["hog_columns"].as<int>(),
                CV_32F,
                (void*)hogBinary.data()
            );

            // std::cout << "after converting mat" << std::endl;
            shoePropertiesList.shoeImageIds.push_back(shoeImageId);

            std::vector<cv::Mat> shoeHistograms;
            // std::cout << "shoeHistograms size: " << shoeHistograms.size() << std::endl;
            shoeHistograms.push_back(redHist.clone());
            // std::cout << "shoeHistograms size: " << shoeHistograms.size() << std::endl;
            shoeHistograms.push_back(greenHist.clone());
            shoeHistograms.push_back(blueHist.clone());
            // std::cout << "shoeHistograms size: " << shoeHistograms.size() << std::endl;

            shoePropertiesList.RGBHistograms.push_back(shoeHistograms);
            // std::cout << "RGBHistograms size: " << shoePropertiesList.RGBHistograms.size() << std::endl;
            shoePropertiesList.LBPHistograms.push_back(lbpFeatures.clone());
            // std::cout << "LBPHistograms size: " << shoePropertiesList.LBPHistograms.size() << std::endl;
            shoePropertiesList.HOGFeatures.push_back(hogFeatures.clone());
            // std::cout << "HOGFeatures size: " << shoePropertiesList.HOGFeatures.size() << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    return shoePropertiesList;
}





void saveShoeColor(int id, ShoeColor shoeColor) {
    std::cout << "Saving shoe properties to database" << std::endl;
    std::cout << "ID: " << id << std::endl;
    std::cout << "Shoe color blue: " << shoeColor.blue << std::endl;

    try {
        if (!conn.is_open()) {
            std::cerr << "Database connection not open" << std::endl;
            return;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return;
    }

    try {
        pqxx::work txn(conn);
        std::cout << "INSERT INTO public.evaluate_shoeproperties (percentage_red,percentage_green,percentage_blue,shoe_image_id) VALUES (" + txn.quote(shoeColor.red) + "," + txn.quote(shoeColor.green) + "," + txn.quote(shoeColor.blue) + "," + txn.quote(id) + ")" << std::endl;

        txn.exec("INSERT INTO public.evaluate_shoeproperties (percentage_red,percentage_green,percentage_blue,shoe_image_id) VALUES (" + txn.quote(shoeColor.red) + "," + txn.quote(shoeColor.green) + "," + txn.quote(shoeColor.blue) + "," + txn.quote(id) + ")");
        txn.commit();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return;
    }
}

void saveDominantColors(int id, std::vector<DominantColor> dominantColors) {
    std::cout << "Saving dominant colors to database" << std::endl;
    std::cout << "ID: " << id << std::endl;

    try {
        if (!conn.is_open()) {
            std::cerr << "Database connection not open" << std::endl;
            return;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return;
    }

    try {
        pqxx::work txn(conn);

        for (DominantColor dominantColor: dominantColors) {
            int red = dominantColor.color[2];
            int green = dominantColor.color[1];
            int blue = dominantColor.color[0];
            // std::cout << "INSERT INTO public.evaluate_shoedominantcolor (red,green,blue,frequency_percentage,shoe_image_id) VALUES (" + txn.quote(red) + "," + txn.quote(green) + "," + txn.quote(blue) + "," + txn.quote(dominantColor.percentage) + "," + txn.quote(id) + ")" << std::endl;

            txn.exec("INSERT INTO public.evaluate_shoedominantcolor (red,green,blue,frequency_percentage,shoe_image_id) VALUES (" + txn.quote(red) + "," + txn.quote(green) + "," + txn.quote(blue) + "," + txn.quote(dominantColor.percentage) + "," + txn.quote(id) + ")");
        }

        txn.commit();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return;
    }
}



// Method to test PostgreSQL connection
void testGetShoeMetadata() {
    try {
        pqxx::work txn(conn);

        for (auto row: txn.exec("SELECT price FROM public.evaluate_shoemetadata")) {
            std::cout << row[0].as<float>() << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}



#endif