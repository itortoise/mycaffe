#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>  // for std:: make_pair
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
////////////////
// opencv dependecies
////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

void convert_dataset (const char* sourcefilename, const char* db_filename){
	std::ifstream infile(sourcefilename);
	std::vector< std::pair<std::string, int> >lines_;
	std::string filename;
	int lebel;
	while(infile>>filename>>lebel){
		lines_.push_back(std::make_pair(filename, lebel));
	}

	//lmdb
	//MDB_env *mdb_env;
	//MDB_dbi mdb_dbi;
	//MDB_val mdb_key, mdb_data;
	//MDB_txn *mdb_txn;
	//leveldb
	leveldb::DB* db;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	options.write_buffer_size = 268435456;
	leveldb::WriteBatch* batch =NULL;
	leveldb::Status status = leveldb::DB::Open(
    options, db_filename, &db);
    CHECK(status.ok()) << "Failed to open leveldb " << db_filename
                       << ". Is it already existing?";
    
    cv::Mat img_ = cv::imread(lines_[0].first);
    const int img_height = img_.rows;
    const int img_width = img_.cols;
    const int num_item = 10000;

    caffe::Datum datum;
    datum.set_channels(2);
    datum.set_height(img_height);
    datum.set_width(img_width);
    const int kMaxKeyLength = 10;
    char key[kMaxKeyLength];
    std::string value;
    LOG(INFO)<<"A total of "<<num_item<<" paris.";
    LOG(INFO)<<"ROWS:"<<img_height<<"COLS:"<<img_width;
    std::string buffer (2*img_height*img_width,' ');
    for(int itemid = 0;itemid<num_item;itemid++){
    	int i = caffe::caffe_rng_rand() % num_item;  // pick a random  pair
    	int j = caffe::caffe_rng_rand() % num_item;
    	std::string filename_i = lines_[i-1].first;
    	std::string filename_j = lines_[i-1].first;
    	int label_i = lines_[i-1].second;
    	int label_j = lines_[j-1].second;
    	cv::Mat img_i = cv::imread(filename_i);
    	cv::Mat img_j = cv::imread(filename_j);
    	std::string buffer (2*img_height*img_width,' ');
    	for (int h = 0;h<img_height;h++){
    		const uchar* ptr = img_i.ptr<uchar>(h);
    		int img_index = 0;
    		for(int w = 0; w < img_width; w++){
    			int buffer_index = w+h*img_width;
    			buffer[buffer_index] = static_cast<char>(ptr[img_index++]);
    		}
    	}
    	for (int h = 0;h<img_height;h++){
    		const uchar* ptr = img_i.ptr<uchar>(h);
    		int img_index = 0;
    		for(int w = 0; w < img_width; w++){
    			int buffer_index = w+h*img_width;
    			buffer[img_height*img_width+buffer_index] = static_cast<char>(ptr[img_index++]);
    		}
    	}
    	datum.set_data(buffer);
    	if (label_j ==label_i){
    		datum.set_label(1);
    	}else{
    		datum.set_label(0);
    	}
    	datum.SerializeToString(&value);
    	snprintf(key, kMaxKeyLength, "%08d", itemid);
    	db->Put(leveldb::WriteOptions(),std::string(key), value);

    }
    delete db;
    delete &buffer;

}
int main(int argc, char** argv){
	if (argc != 3) {
    printf("This script converts the MNIST dataset to the leveldb format used\n"
           "by caffe to train a siamese network.\n"
           "Usage:\n"
           "    convert_mnist_data input_image_file input_label_file "
           "output_db_file\n"
           "The MNIST dataset could be downloaded at\n"
           "    http://yann.lecun.com/exdb/mnist/\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(argv[1], argv[2]);
  }
  return 0;
}