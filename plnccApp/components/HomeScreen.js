import React from 'react';
import {Platform, TouchableOpacity,View, Text, StyleSheet} from 'react-native';
import CircleButton from 'react-native-circle-button';
import {Calendar} from 'react-native-calendars';
import { SwitchNavigator, createSwitchNavigator, createStackNavigator, createAppContainer} from 'react-navigation'

class HomeScreen extends React.Component {
  render() {
    const {navigate} = this.props.navigation;
    return (
    <View style={style.bigContainer}>
      <View style={style.calendarContainer}>
        <Calendar
          style={style.calendarSpecs}
          onDayPress={() => {
              navigate('ResultsPage')
            }}
        />
      </View>
      <View style={{ flex: 1 }}>
        <TouchableOpacity style={style.btn}
        onPress = {() => {
            navigate('CameraPage')
          }}>
          <Text style= {{fontSize: 40}}>+</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};
}

const style = StyleSheet.create({
  bigContainer: {
    padding: 60,
  },
  calendarContainer: {
    height: '80%',
    borderColor: 'black'
  },
  calendarSpecs: {
    borderWidth: 1,
    borderColor: "#ccc",
  },
  listItem: {
    padding: 10,
    marginVertical: 2,
    backgroundColor: '#ccc',
    borderColor: 'black',
    borderWidth: 1
  },
  button: {
    padding: 5,
    height: '20%',
    width: '20%',  //The Width must be the same as the height
    borderRadius:400, //Then Make the Border Radius twice the size of width or Height
    borderColor: 'black',
    backgroundColor:'#ccc',
  },
btn:{
  position: 'absolute',
  width: 100, height: 100,
  backgroundColor: '#9acbe6',
  borderRadius:50,
  borderColor: 'black',
  alignItems: 'center',
  alignSelf: 'center',
  justifyContent: 'center',
},
});

export default HomeScreen;
