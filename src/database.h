#ifndef DATABASE_H
#define DATABASE_H

#include <iostream>
#include "evaluate.h"

std::string connString = "host=winhost port=5432 dbname=shoes user=postgres password=root";
pqxx::connection conn(connString.c_str());

void saveShoeProperties(int id, ShoeColor shoeColor) {
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
        std::cout << "INSERT INTO public.evaluate_shoeproperties (percentage_red,percentage_green,percentage_blue,shoe_id) VALUES (" + txn.quote(shoeColor.red) + "," + txn.quote(shoeColor.green) + "," + txn.quote(shoeColor.blue) + "," + txn.quote(id) + ")" << std::endl;

        txn.exec("INSERT INTO public.evaluate_shoeproperties (percentage_red,percentage_green,percentage_blue,shoe_id) VALUES (" + txn.quote(shoeColor.red) + "," + txn.quote(shoeColor.green) + "," + txn.quote(shoeColor.blue) + "," + txn.quote(id) + ")");
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