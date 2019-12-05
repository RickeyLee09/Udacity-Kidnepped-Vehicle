/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
    num_particles = 101;  // TODO: Set the number of particles

    std::default_random_engine gen;

    // TODO: Create normal distributions for y and theta
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {
        Particle p;

        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;

        particles.push_back(p);
    }

    is_initialized = true;
    return;

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

    std::default_random_engine gen;

    std::normal_distribution<double> dist_x(0, std_pos[0]);
    std::normal_distribution<double> dist_y(0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {
        if(yaw_rate != 0.0){
            particles[i].x += (velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta))) + dist_x(gen);
            particles[i].y += (velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t))) + dist_y(gen);
            particles[i].theta += (yaw_rate * delta_t) + dist_theta(gen);
        }else{
            particles[i].x += (velocity * delta_t * cos(particles[i].theta)) + dist_x(gen);
            particles[i].y += (velocity * delta_t * sin(particles[i].theta)) + dist_y(gen);
            particles[i].theta += (yaw_rate * delta_t) + dist_theta(gen);
        }

    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

    for(int i =0; i<observations.size(); i++){
        LandmarkObs obs = observations[i];

        double min_dist = 100000000;
        int map_id = -1;

        for(int j=0; j<predicted.size(); j++){
            LandmarkObs predict = predicted[j];

            double distance = dist(obs.x, obs.y, predict.x, predict.y);

            if(distance < min_dist){
                min_dist = distance;
                map_id = predict.id;
            }
        }

        observations[i].id = map_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

    for(int i=0; i<num_particles; i++){
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        std::vector<LandmarkObs> landmarks;

        for(int j=0; j<map_landmarks.landmark_list.size(); j++){
            float landmark_x = map_landmarks.landmark_list[j].x_f;
            float landmark_y = map_landmarks.landmark_list[j].y_f;
            int landmark_id = map_landmarks.landmark_list[j].id_i;

//            if(sqrt(pow((landmark_x - p_x), 2)) + pow((landmark_y - p_y), 2) <= sensor_range){
            if(fabs(landmark_x - p_x) <= sensor_range && fabs(landmark_y - p_y) <= sensor_range){
                landmarks.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
            }
        }

        std::vector<LandmarkObs> tobs;

        for(int j=0; j<observations.size(); j++){
            int t_id = observations[j].id;
            double t_x = p_x + (cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y);
            double t_y = p_y + (sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y);

            tobs.push_back(LandmarkObs{t_id, t_x, t_y});
        }

        dataAssociation(landmarks, tobs);

        particles[i].weight = 1.0;

        for(int j=0; j<tobs.size(); j++){
            double tob_x, tob_y, predict_x, predict_y;

            tob_x = tobs[j].x;
            tob_y = tobs[j].y;

            int ass_id = tobs[j].id;

            for(int k=0; k<landmarks.size(); k++){
                if(landmarks[k].id == ass_id){
                    predict_x = landmarks[k].x;
                    predict_y = landmarks[k].y;
                }
            }

            double std_x = std_landmark[0];
            double std_y = std_landmark[1];


            double a = pow(predict_x - tob_x, 2) / (2 * pow(std_x, 2)) + (pow(predict_y - tob_y, 2)/(2 * pow(std_y, 2)));
            double b = exp(-a);
            double weight = (1 / (2 * M_PI * std_x * std_y)) * b;

            particles[i].weight *= weight;

        }
    }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

    std::vector<Particle> new_particles;

    std::vector<double> weights;

    for(int i=0; i<num_particles; i++){
        weights.push_back(particles[i].weight);
    }

    std::uniform_int_distribution<int> dist(0, num_particles-1);
    std::default_random_engine gen;
    auto index = dist(gen);

    double max_weight = *max_element(weights.begin(), weights.end());

    std::uniform_real_distribution<double> dist_real(0.0, max_weight);
    double beta = 0.0;

    for(int i=0; i<num_particles; i++){
        beta += dist_real(gen) * 2.0;

        while(beta>weights[index]){
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }

        new_particles.push_back(particles[index]);
    }

    particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const std::vector<int>& associations,
                                     const std::vector<double>& sense_x,
                                     const std::vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  std::vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  std::vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}