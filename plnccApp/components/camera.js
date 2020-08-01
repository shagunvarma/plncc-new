import React, { Component } from 'react';
import {
  AppRegistry,
  StyleSheet,
  Text,
  View
} from 'react-native';

import Camera from 'react-native-camera'

export default class cameraApp extends React.Component {
  render() {
    return (
      <View style={styles.container}>
        <Camera
          ref={(cam) => {
            this.camera = cam
          }}
          style = {styles.view}
          aspect = {Camera.constants.Aspect.fill}
            <Button
            style = {styles.capture}
            onpress = {this.takePicture.bind(this)}>
              [CAPTURE_IMAGE]/>
        </Camera>
    </View>
  );
}

  takePicture() {
    const options = {}

    this.camera.capture({metadata : options}).then((data) => {
      console.log(data)
    }).catch((error) => {
      console.log(error)
    })
  }
}

const styles = StyleSheet.create({
    containter: {
      flex: 1,
      flexDirection: 'row'
    },
    view: {
      flex: 1,
      justifyContent: 'flex-end',
      alignItems: 'center'
    }
    capture: {
      flex:0,
      backgroundColor: 'steelblue',
      borderRadious: 10,
      color: 'red',
      padding: 15,
      margin: 45
    }
});

AppRegistry.registerComponent('cameraApp', () => cameraApp);
